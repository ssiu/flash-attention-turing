#!/usr/bin/env python3
"""Portable D256 native O/LSE CUDA-graph benchmark; no lifecycle control.

--help and --cpu-check do not import torch or initialize CUDA. GPU execution
is explicit. Native and canonical output scopes are reported separately.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace


TESTED_CODE_BASE = "1b777e5db301b9ca5815e4910fd14576f8c4ae0e"
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def useful_flops(q, k, heads=12, dim=256, batch=1):
    """QK + PV only; exact append-causal useful pairs, no softmax FLOPs."""
    if not 0 < q <= k:
        raise ValueError("Require 0 < Q <= KV")
    pairs = q * (k - q) + q * (q + 1) // 2
    return 4 * batch * heads * dim * pairs


def stats(values):
    median = statistics.median(values)
    return dict(raw_ms=values, median_ms=median,
                mad_ms=statistics.median(abs(x - median) for x in values),
                min_ms=min(values), max_ms=max(values), mean_ms=statistics.mean(values))


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def error_metrics(torch, value, reference, dtype, *, promoted_candidate=False):
    """NRMS matches the existing .001 O screen; ULP is diagnostic only."""
    value = value.detach().cpu()
    reference = reference.detach().cpu().double()
    expected_dtype = torch.float64 if promoted_candidate and dtype == torch.float32 else dtype
    if value.shape != reference.shape or value.dtype != expected_dtype:
        raise ValueError("Metric shape/dtype mismatch")
    if not bool(torch.isfinite(value).all() and torch.isfinite(reference).all()):
        return dict(finite=False)
    error = (value.double() - reference).abs().flatten()
    rms = float(error.square().mean().sqrt())
    width, integer = (16, torch.int16) if dtype == torch.float16 else (32, torch.int32)
    def ordered(x):
        bits = x.to(dtype).contiguous().view(integer).to(torch.int64)
        magnitude = bits & ((1 << (width - 1)) - 1)
        return torch.where(bits < 0, (1 << (width - 1)) - magnitude,
                           (1 << (width - 1)) + magnitude)
    rounded = reference.to(dtype)
    if not bool(torch.isfinite(rounded).all()):
        return dict(finite=False, rounded_reference_nonfinite=True)
    ulp = (ordered(value) - ordered(rounded)).abs().flatten()
    def percentile(x, fraction):
        return float(x.kthvalue(math.ceil(x.numel() * fraction)).values)
    return dict(finite=True, elements=error.numel(), rms=rms,
                normalized_rms=rms / max(float(reference.square().mean().sqrt()), 1e-12),
                raw_mean_abs=float(error.mean()), raw_max_abs=float(error.max()),
                raw_abs_p99=percentile(error, .99), raw_abs_p999=percentile(error, .999),
                ulp_p99=int(percentile(ulp, .99)), ulp_p999=int(percentile(ulp, .999)),
                ulp_max=int(ulp.max()))


def fp64_reference(torch, qkv, rows, scale):
    """Bounded CPU streaming oracle, all heads, key chunks4096, rows<=7."""
    q, k, v = qkv
    b, nq, hq, dim = q.shape
    nk, hkv = k.shape[1:3]
    if len(rows) > 7 or any(not 0 <= x < nq for x in rows):
        raise ValueError("Bounded reference rows required")
    outputs = torch.empty((b, len(rows), hq, dim), dtype=torch.float64)
    lse = torch.empty((b, hq, len(rows)), dtype=torch.float64)
    with torch.no_grad():
        for ib in range(b):
            for h in range(hq):
                kh = h // (hq // hkv)
                qq = q[ib, rows, h].double()
                m = torch.full((len(rows),), -math.inf, dtype=torch.float64)
                denominator = torch.zeros_like(m)
                numerator = torch.zeros((len(rows), dim), dtype=torch.float64)
                for start in range(0, nk, 4096):
                    stop = min(start + 4096, nk)
                    scores = qq @ k[ib, start:stop, kh].double().T * scale
                    scores.masked_fill_(torch.arange(start, stop)[None, :] >
                                        (nk - nq + torch.tensor(rows))[:, None], -math.inf)
                    new_m = torch.maximum(m, scores.max(dim=1).values)
                    alpha = torch.exp(m - new_m)
                    probability = torch.exp(scores - new_m[:, None])
                    numerator = numerator * alpha[:, None] + probability @ v[ib, start:stop, kh].double()
                    denominator = denominator * alpha + probability.sum(dim=1)
                    m = new_m
                outputs[ib, :, h] = numerator / denominator[:, None]
                lse[ib, h] = m + denominator.log()
    return outputs, lse


def inputs(torch, q, k, args):
    shapes = ((1, q, 12, 256), (1, k, 2, 256), (1, k, 2, 256))
    seed = args.seed + q * 131 + k * 17
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cpu = tuple((torch.randn(shape, generator=generator) * .7).half().contiguous()
                for shape in shapes)
    return cpu, dict(label="seeded_synthetic_remeasurement", seed=seed,
                     input_sha256=[tensor_hash(x) for x in cpu],
                     recipe="CPU sequential torch.randn(Q,K,V), multiply0.7 in FP32, cast FP16, contiguous",
                     shapes=[list(x.shape) for x in cpu])


def module_identity(module):
    path = Path(module.__file__).resolve()
    return dict(module=module.__name__, file=str(path), sha256=sha(path))


def loaded_fi_binaries():
    """Linux evidence of actual loaded FI/JIT shared objects, after dispatch."""
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return dict(available=False, reason="No /proc/self/maps; publish dispatch provenance separately")
    paths = set()
    for line in maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and "flashinfer" in fields[5].lower():
            path = Path(fields[5])
            if path.is_file() and ".so" in path.name:
                paths.add(path.resolve())
    return dict(available=True, binaries={str(p): sha(p) for p in sorted(paths)},
                limit="Mapped FI shared-object identities; not a full compiler/runtime supply-chain audit")


def fi_identity(module, args):
    """Require FI0.7; verify explicitly pinned sources if build metadata is missing."""
    version = module.__version__
    evidence = dict(**module_identity(module), version=version,
                    reported_git_commit=getattr(module, "__git_commit__", None))
    if args.fi_source_manifest:
        path = Path(args.fi_source_manifest)
        if sha(path) != args.fi_source_manifest_sha256:
            raise ValueError("FI source manifest SHA differs")
        doc = json.loads(path.read_text())
        if doc["source"]["source_version_txt"] != "0.7.0":
            raise ValueError("FI source commit/version differs")
        package = Path(module.__file__).resolve().parent
        python_files = sorted(package.rglob("*.py"))
        tree = hashlib.sha256("".join(f"{p.relative_to(package.parent)}:{sha(p)}\n" for p in python_files).encode()).hexdigest()
        if (len(python_files) != doc["target"]["package_python_file_count"]
                or tree != doc["target"]["package_python_tree_sha256"]):
            raise ValueError("Actual imported FI package does not match pinned0.7 Python tree")
        header = package / "data/include/flashinfer/attention/prefill.cuh"
        expected_header = doc["source"]["key_source_sha256"]["include/flashinfer/attention/prefill.cuh"]
        if not header.is_file() or sha(header) != expected_header:
            raise ValueError("Actual FI prefill header differs")
        if version not in ("0.0.0+unknown", "0.7.0"):
            raise ValueError("FI build metadata unexpectedly names another version")
        evidence.update(source_commit=doc["source"]["source_commit"], source_manifest_sha256=sha(path),
                        actual_python_tree_sha256=tree, prefill_header_sha256=expected_header,
                        acceptance="Explicitly pinned0.7 source identity; missing build meta is not another-version exemption")
    elif version.split("+")[0] == "0.7.0":
        evidence["acceptance"] = "Official reported0.7.0; source/JIT binary provenance recorded separately"
    else:
        raise ValueError("Require FI0.7.0 or an explicitly pinned0.7 source manifest/tree, not unrelated versions")
    if args.fi_binary:
        binary = Path(args.fi_binary).resolve()
        if sha(binary) != args.fi_binary_sha256:
            raise ValueError("FI binary SHA differs")
        import flashinfer.prefill as prefill
        import tvm_ffi
        compiled = tvm_ffi.load_module(str(binary))
        def prebuilt_only(backend, *parameters):
            if backend != "fa2" or prefill.get_single_prefill_uri(backend, *parameters) != binary.stem:
                raise ValueError("Prebuilt FI URI/backend differs; no JIT fallback")
            return SimpleNamespace(build_and_load=lambda: compiled)
        prefill.gen_single_prefill_module = prebuilt_only
        evidence.update(binary=str(binary), binary_sha256=args.fi_binary_sha256,
                        dispatch="Explicit prebuilt-only fa2 URI check; no JIT fallback")
    else:
        evidence["dispatch"] = "backend=fa2; optional first-use JIT outside Events"
    return evidence


def calibrate_fi(torch, module, device):
    """Untimed hand-computable append-causal O/log-base check, before corpus."""
    q = torch.zeros((3, 12, 256), device=device, dtype=torch.float16)
    k = torch.zeros((7, 2, 256), device=device, dtype=torch.float16)
    v = torch.full_like(k, .5)
    with torch.no_grad():
        o, raw_lse = module.single_prefill_with_kv_cache(
            q, k, v, causal=True, sm_scale=.0625, backend="fa2", return_lse=True)
    if tuple(o.shape) != (3, 12, 256) or tuple(raw_lse.shape) != (3, 12):
        raise ValueError("FI calibration output contract differs")
    if not bool(torch.isfinite(o).all() and torch.isfinite(raw_lse).all()):
        raise ValueError("FI calibration nonfinite")
    expected = torch.tensor([math.log(x) for x in (5, 6, 7)], dtype=torch.float64)
    error = float((raw_lse.detach().cpu().double() * math.log(2.0) - expected[:, None]).abs().max())
    if not torch.equal(o, torch.full_like(o, .5)) or error > 1e-5:
        raise ValueError("FI zero-QK O/log2 convention check failed before corpus")
    return dict(valid_counts=[5, 6, 7], O_constant=.5, promoted_ln_max_abs=error,
                calibration_bound=1e-5, corpus_LSE_gate=False, timing=False)


def source_identity():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
    files = [Path(__file__).resolve(), root / "setup.py", root / "flash_attention_interface.py"]
    files += list((root / "csrc/flash_attn").glob("*d256*"))
    files += list((root / "csrc/flash_attn/src").glob("*d256*"))
    files += list((root / "csrc/flash_attn/src").glob("*hdim256*"))
    files += [root / "csrc/flash_attn/src" / name for name in
              ("utils.h", "mask.h", "block_info.h", "static_switch.h", "flash.h", "kernel_traits.h")]
    return dict(tested_code_base=TESTED_CODE_BASE, actual_commit=git("rev-parse", "HEAD"),
                worktree_status=git("status", "--short"),
                cutlass_commit=git("rev-parse", "HEAD:csrc/cutlass"),
                files={str(p.relative_to(root)): sha(p) for p in files if p.is_file()})


def validate_trace(doc, arm, scope, q, k):
    kernels = [x for x in doc["traceEvents"] if x.get("cat") == "kernel"]
    if arm == "FAT":
        if len(kernels) != 1:
            raise ValueError("Native FAT must emit one attention kernel, no manual fills")
        event = kernels[0]
        name = event["name"]
        if not re.search(r"(?<![A-Za-z0-9_])flash_fwd_kernel(?:_original_grid)?<", name):
            raise ValueError("Unknown FAT entry")
        attrs = event["args"]
        m = (q + 63) // 64
        original = "flash_fwd_kernel_original_grid<" in name
        grid = [m, 1, 12] if original else [12, m, 1]
        if (attrs.get("grid") != grid or attrs.get("block", attrs.get("blockDim")) != [256, 1, 1]
                or int(attrs.get("shared memory", attrs.get("sharedMemory", -1))) != 58112):
            raise ValueError("FAT launch contract differs")
        even = q % 64 == 0 and k % 32 == 0
        if "Flash_fwd_kernel_traits<256, 64, 32, 4," not in name or f">, true, {str(even).lower()}>" not in name:
            raise ValueError("FAT specialization differs")
        conversion = []
    else:
        attention = [x for x in kernels if "SinglePrefillWithKVCacheKernel<" in x["name"]]
        merge = [x for x in kernels if "MergeStates" in x["name"] and "Kernel<" in x["name"]]
        conversion = [x for x in kernels if x not in attention + merge]
        if len(attention) != 1 or len(merge) > 1 or len(conversion) != (1 if scope == "canonical" else 0):
            raise ValueError("Unexpected FI split/merge/conversion trace; inspect rather than silently broaden")
        if q == 8192 and k in (65536, 131072):
            attrs = attention[0]["args"]
            if (merge or attrs.get("grid") != [768, 1, 2]
                    or attrs.get("block", attrs.get("blockDim")) != [32, 4, 1]
                    or int(attrs.get("shared memory", attrs.get("sharedMemory", -1))) != 65536):
                raise ValueError("Frozen long FI dispatch differs; do not claim timing parity")
        if conversion and not ("elementwise_kernel" in conversion[0]["name"] and "MulFunctor<float>" in conversion[0]["name"]):
            raise ValueError("Unexpected canonical conversion")
    return dict(kernel_count=len(kernels), kernels=kernels,
                conversion_count=len(conversion), no_manual_fills=True)


def capture(torch, call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        eager = call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        values = call()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    if len(values) != 2 or not all(bool(torch.isfinite(x).all()) for x in values):
        raise ValueError("Nonfinite O/LSE; no timing")
    if not all(torch.equal(a.detach().contiguous().view(torch.uint8), b.detach().contiguous().view(torch.uint8))
               for a, b in zip(eager, values)):
        raise ValueError("Eager/graph mismatch")
    return graph, values


def trace(torch, graph, path, arm, scope, q, k):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                          torch.profiler.ProfilerActivity.CUDA]) as profiler:
        graph.replay()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    return validate_trace(json.loads(path.read_text()), arm, scope, q, k)


class ClockSampler:
    """Read-only optional1Hz snapshots; no clock/power/process changes."""
    def __init__(self, uuid):
        self.uuid, self.rows, self.errors = uuid, [], []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                query = "timestamp,name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,pcie.link.width.current"
                row = subprocess.check_output(["nvidia-smi", "--id=" + self.uuid,
                                               "--query-gpu=" + query, "--format=csv,noheader,nounits"],
                                              text=True, timeout=5).strip()
                self.rows.append(dict(monotonic_ns=time.monotonic_ns(), raw=row))
            except (OSError, subprocess.SubprocessError) as error:
                self.errors.append(str(error))
            self.stop.wait(1)


def timed_pair(torch, graphs, arms, out, scope, q, k, uuid):
    for _ in range(5):
        for arm in arms:
            graphs[arm].replay()
    torch.cuda.synchronize()
    samples = {a: [] for a in arms}
    orders = []
    sampler = ClockSampler(uuid)
    sampler.thread.start()
    started = time.monotonic_ns()
    try:
        # No oracle, hashes, profiler or health queries in this Event loop.
        for iteration in range(30):
            order = arms if iteration % 2 == 0 else arms[::-1]
            orders.append(list(order))
            for arm in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[arm].replay()
                end.record()
                end.synchronize()
                samples[arm].append(start.elapsed_time(end))
    finally:
        ended = time.monotonic_ns()
        sampler.stop.set()
        sampler.thread.join(timeout=6)
    result = dict(samples_ms=samples, orders=orders, warmup=5, pairs=30,
                  event_bracket_monotonic_ns=[started, ended],
                  timing={a: stats(x) for a, x in samples.items()},
                  useful_flops=useful_flops(q, k), clock_samples=sampler.rows,
                  clock_errors=sampler.errors, clock_scope="1Hz external sampler; not per-arm clocks",
                  scope=scope, timing_scope="CUDA graph replay of native call; Python, input copies and oracle excluded")
    result["useful_tflops"] = {a: result["useful_flops"] / stats(samples[a])["median_ms"] / 1e9 for a in arms}
    result["ratio_comparator_over_FAT"] = statistics.median(samples[arms[1]]) / statistics.median(samples["FAT"])
    result["wins_FAT"] = sum(a < b for a, b in zip(samples["FAT"], samples[arms[1]]))
    result["half_ratios_comparator_over_FAT"] = [statistics.median(samples[arms[1]][s:s+15]) / statistics.median(samples["FAT"][s:s+15]) for s in (0, 15)]
    write(out / (scope + "-raw.json"), result)
    result["traces"] = {a: trace(torch, graphs[a], out / (scope + "-" + a + ".trace.json"), a, scope, q, k) for a in arms}
    write(out / (scope + "-result.json"), result)
    return result


def run(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    completed = []
    write(out / "invocation.json", dict(arguments=vars(args), source=source_identity(),
                                       public_driver_GPU_validation="not implied by source preparation"))
    try:
        import torch
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_device(args.device)
        if torch.cuda.get_device_capability() != (7, 5):
            raise ValueError("SM75 required")
        prop = torch.cuda.get_device_properties(args.device)
        uuid = str(prop.uuid)
        if not uuid.startswith("GPU-"):
            uuid = "GPU-" + uuid
        fat = importlib.import_module("flash_attn_turing_d256")
        modules = {"FAT": module_identity(fat)}
        if args.fi_python_root:
            if "flashinfer" in sys.modules:
                raise ValueError("FI imported before explicit source-root selection")
            sys.path.insert(0, str(Path(args.fi_python_root).resolve()))
        comparator = importlib.import_module("flashinfer")
        modules["FI"] = fi_identity(comparator, args)
        write(out / "FI-log-base-calibration.json", calibrate_fi(torch, comparator, f"cuda:{args.device}"))
        write(out / "runtime.json", dict(torch=torch.__version__, cuda=torch.version.cuda,
                                         python=sys.version, device=args.device, gpu_name=prop.name,
                                         capability=[prop.major, prop.minor], total_memory=prop.total_memory,
                                         module_identity=modules))
        cells = ((args.q, args.kv),)
        with torch.no_grad():
            for q, k in cells:
                cell = out / f"q{q}-kv{k}"
                cell.mkdir()
                cpu, provenance = inputs(torch, q, k, args)
                rows = sorted({0, q // 2, q - 1})
                reference, reference_lse = fp64_reference(torch, cpu, rows, .0625)
                qv, kv, vv = (x.to(device=f"cuda:{args.device}") for x in cpu)
                fat_call = lambda: fat.forward(qv, kv, vv, .0625, True)
                arm = "FI"
                other_call = lambda: comparator.single_prefill_with_kv_cache(qv[0], kv[0], vv[0], causal=True, sm_scale=.0625, backend="fa2", return_lse=True)
                graphs, values = {}, {}
                for name, call in (("FAT", fat_call), (arm, other_call)):
                    graphs[name], values[name] = capture(torch, call)
                expected_shapes = {"FAT": [(1, q, 12, 256), (1, 12, q)],
                                   arm: [(q, 12, 256), (q, 12)]}
                for name in values:
                    if [tuple(x.shape) for x in values[name]] != expected_shapes[name] or values[name][0].dtype != torch.float16 or values[name][1].dtype != torch.float32:
                        raise ValueError("Native output shape/dtype differs")
                if not values["FAT"][1].is_contiguous():
                    raise ValueError("FAT canonical LSE must already be contiguous")
                metrics = {}
                for name, (o, lse) in values.items():
                    o_sample = o[rows].unsqueeze(0) if name == "FI" else o[:, rows]
                    natural = (lse.double().T.unsqueeze(0) * math.log(2.0) if name == "FI" else lse.double())
                    metrics[name] = dict(O=error_metrics(torch, o_sample, reference, torch.float16),
                                         natural_ln_LSE=error_metrics(torch, natural[:, :, rows], reference_lse, torch.float32, promoted_candidate=True),
                                         promoted_LSE_max_abs=float((natural[:, :, rows].cpu() - reference_lse).abs().max()))
                write(cell / "numeric.json", dict(metrics=metrics, provenance=provenance, rows=rows,
                                                   scale_requested=.0625, scale_API_FP32=.0625,
                                                   reference="CPU FP64 key_chunk4096, fixed rows/all heads",
                                                   FI_is_not_truth=True, LSE_ULP_diagnostic_only=True))
                write(cell / "FI-loaded-binaries.json", loaded_fi_binaries())
                if not metrics["FAT"]["O"]["finite"] or metrics["FAT"]["O"]["normalized_rms"] > .001:
                    raise ValueError("Bounded FAT NRMS .001 screen failed; no timing")
                results = {}
                if args.scope in ("native", "both"):
                    results["native"] = timed_pair(torch, graphs, ("FAT", arm), cell, "native", q, k, uuid)
                if args.scope in ("canonical", "both"):
                    natural_out = torch.empty((1, 12, q), device=qv.device, dtype=torch.float32)
                    raw_state = {}
                    def canonical_call():
                        o, raw_lse = other_call()
                        raw_state["lse"] = raw_lse
                        torch.mul(raw_lse.transpose(0, 1), math.log(2.0), out=natural_out[0])
                        return o.unsqueeze(0), natural_out
                    canonical_graph, canonical_value = capture(torch, canonical_call)
                    expected = (raw_state["lse"].detach().cpu().T * math.log(2.0)).unsqueeze(0)
                    actual = canonical_value[1].detach().cpu()
                    if not actual.is_contiguous() or not torch.equal(actual.view(torch.int32), expected.contiguous().view(torch.int32)):
                        raise ValueError("Canonical FP32 LSE conversion differs")
                    conversion = dict(GPU_FP32_mul_matches_CPU_FP32_bits=True,
                                      max_abs_to_promoted_double=float((actual.double() - raw_state["lse"].detach().cpu().double().T.unsqueeze(0) * math.log(2.0)).abs().max()),
                                      numeric_vs_FP64=error_metrics(torch, actual[:, :, rows], reference_lse, torch.float32))
                    write(cell / "canonical-conversion.json", conversion)
                    results["canonical"] = timed_pair(torch, {"FAT": graphs["FAT"], arm: canonical_graph}, ("FAT", arm), cell, "canonical", q, k, uuid)
                completed.append(dict(q=q, kv=k, ratios={s: v["ratio_comparator_over_FAT"] for s, v in results.items()}))
                del graphs, values, qv, kv, vv
                torch.cuda.synchronize()
        summary = dict(completed=True, cells=completed,
                       reference_scope="sampled rows, not full-output qualification")
        write(out / "summary.json", summary)
    except BaseException as error:
        write(out / "failed.json", dict(error=repr(error), traceback=traceback.format_exc(), completed=completed))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-check", action="store_true", help="stdlib-only policy check; no Torch/CUDA")
    parser.add_argument("--device", type=int, help="explicit CUDA logical index after CUDA_VISIBLE_DEVICES")
    parser.add_argument("--out", help="fresh output directory")
    parser.add_argument("--q", type=int, default=8192)
    parser.add_argument("--kv", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=2026100801)
    parser.add_argument("--fi-python-root", help="optional explicit0.7 package-parent directory")
    parser.add_argument("--fi-source-manifest", help="pinned0.7 source manifest for missing build metadata")
    parser.add_argument("--fi-source-manifest-sha256")
    parser.add_argument("--fi-binary", help="optional prebuilt-only fa2 module, no JIT fallback")
    parser.add_argument("--fi-binary-sha256")
    parser.add_argument("--scope", choices=("native", "canonical", "both"), default="both")
    parser.add_argument("--warmup", type=int, choices=(5,), default=5, help="fixed qualified scope")
    parser.add_argument("--samples", type=int, choices=(30,), default=30, help="fixed pairs per scope")
    args = parser.parse_args()
    if args.cpu_check:
        assert useful_flops(1, 1) == 4 * 12 * 256
        print(json.dumps(dict(status="CPU_POLICY_ONLY_PASS", GPU_executed=False,
                              warmup=5, pairs=30, manual_fills=0, tested_code_base=TESTED_CODE_BASE)))
        return
    if args.device is None or args.device < 0 or not args.out:
        parser.error("GPU execution requires --device>=0 and a fresh --out")
    if not 0 < args.kv <= 131072 or args.q <= 0 or args.q > min(args.kv, 8192):
        parser.error("Require 0<Q<=min(KV,8192), 0<KV<=131072 for this bounded driver")
    if bool(args.fi_source_manifest) != bool(args.fi_source_manifest_sha256) or bool(args.fi_binary) != bool(args.fi_binary_sha256):
        parser.error("FI manifest/binary options require their SHA fields")
    run(args)


if __name__ == "__main__":
    main()
