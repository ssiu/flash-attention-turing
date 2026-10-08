"""Bounded synthetic GPU/API regression, opt-in; not the private94/135 corpus.

SM75_RUN_GPU_TESTS=1 SM75_TEST_DEVICE=<logical index> pytest -q this_file
Ordinary CPU collection runs only the pure policy/metric tests below.
"""
import importlib.util
import json
import math
import os
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "sm75_public_benchmark", Path(__file__).parents[1] / "benchmarks/benchmark_sm75_d256.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
GPU = pytest.mark.skipif(os.environ.get("SM75_RUN_GPU_TESTS") != "1",
                         reason="GPU tests require explicit opt-in and selected device")
# Four actual routing/parity classes and additional rectangular/batch/GQA coverage.
CASES = ((1, 1, 33, 4, 2), (1, 64, 64, 4, 2),
         (1, 65, 129, 6, 1), (1, 128, 160, 4, 2),
         (2, 17, 49, 4, 1), (1, 127, 160, 2, 2))


@pytest.fixture(scope="module")
def runtime():
    # Called only by opted-in GPU tests. No CUDA availability probe at collection.
    selected = os.environ.get("SM75_TEST_DEVICE")
    if selected is None or not selected.isdecimal():
        pytest.fail("Set explicit SM75_TEST_DEVICE logical CUDA index")
    torch.cuda.set_device(int(selected))
    assert torch.cuda.get_device_capability() == (7, 5), "SM75 required"
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    import flash_attn_turing_d256 as native
    import flash_attention_interface as public
    return torch.device("cuda", int(selected)), native, public


def cpu_inputs(case):
    b, q, k, hq, hkv = case
    generator = torch.Generator().manual_seed(2026100800 + q * 131 + k * 17 + b)
    return tuple((torch.randn(shape, generator=generator) * .7).half()
                 for shape in ((b, q, hq, 256), (b, k, hkv, 256), (b, k, hkv, 256)))


def compare(value, lse, cpu, scale, record_property):
    q = cpu[0].shape[1]
    reference, reference_lse = bench.fp64_reference(torch, cpu, list(range(q)), scale) if q <= 7 else full_small_reference(cpu, scale)
    om = bench.error_metrics(torch, value, reference, torch.float16)
    lm = bench.error_metrics(torch, lse, reference_lse, torch.float32)
    record_property("numeric", json.dumps(dict(O=om, natural_ln_LSE=lm,
                                               requested_scale=scale,
                                               effective_FP32_scale=float(torch.tensor(scale, dtype=torch.float32)),
                                               LSE_ULP_diagnostic_only=True)))
    assert om["finite"] and lm["finite"]
    # Existing absolute O NRMS screen only. No FI-win, inherited native-ratio,
    # bitwise inter-backend, arbitrary LSE or ULP numerical PR requirement.
    assert om["normalized_rms"] <= .001, om


def full_small_reference(cpu, scale):
    q, k, v = cpu
    b, nq, hq, d = q.shape
    nk, hkv = k.shape[1:3]
    assert nq <= 128 and nk <= 160, "This full CPU oracle is intentionally bounded"
    out = torch.empty((b, nq, hq, d), dtype=torch.float64)
    lse = torch.empty((b, hq, nq), dtype=torch.float64)
    with torch.no_grad():
        for ib in range(b):
            for h in range(hq):
                kh = h // (hq // hkv)
                score = q[ib, :, h].double() @ k[ib, :, kh].double().T * scale
                score.masked_fill_(torch.arange(nk)[None, :] >
                                   (nk - nq + torch.arange(nq))[:, None], -math.inf)
                out[ib, :, h] = score.softmax(-1) @ v[ib, :, kh].double()
                lse[ib, h] = score.logsumexp(-1)
    return out, lse


def offset_view(value, offset):
    storage = torch.empty(value.numel() + offset, device=value.device, dtype=value.dtype)
    view = storage[offset:].view(value.shape)
    view.copy_(value)
    assert view.is_contiguous() and view.storage_offset() == offset
    return view


@GPU
@pytest.mark.parametrize("case", CASES)
def test_full_small_FP64_and_public_API(runtime, case, record_property):
    device, native, public = runtime
    cpu = cpu_inputs(case)
    values = tuple(x.to(device) for x in cpu)
    with torch.no_grad():
        out, lse = native.forward(*values, .0625, True)
        public_out = public.flash_attn_func(*values, causal=True)
    assert out.shape == cpu[0].shape and out.dtype == torch.float16
    assert lse.shape == (case[0], case[3], case[1]) and lse.dtype == torch.float32
    assert torch.equal(public_out, out), "Same implementation/public routing equality, not PR backend gate"
    compare(out.cpu(), lse.cpu(), cpu, .0625, record_property)


@GPU
@pytest.mark.parametrize("q,k", ((1, 33), (64, 64), (65, 129), (128, 160)))
def test_actual_four_routes(runtime, q, k, tmp_path, record_property):
    device, native, _ = runtime
    values = tuple(x.to(device) for x in cpu_inputs((1, q, k, 4, 2)))
    with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as profiler:
        native.forward(*values, .0625, True)
        torch.cuda.synchronize(device)
    path = tmp_path / "trace.json"
    profiler.export_chrome_trace(str(path))
    kernels = [e for e in json.loads(path.read_text())["traceEvents"] if e.get("cat") == "kernel"]
    assert len(kernels) == 1
    event = kernels[0]
    original = (q + 63) // 64 == 1
    name = "flash_fwd_kernel_original_grid<" if original else "flash_fwd_kernel<"
    even = q % 64 == 0 and k % 32 == 0
    assert name in event["name"] and f">, true, {str(even).lower()}>" in event["name"]
    attrs = event["args"]
    m = (q + 63) // 64
    assert attrs["grid"] == ([m, 1, 4] if original else [4, m, 1])
    assert attrs.get("block", attrs.get("blockDim")) == [256, 1, 1]
    assert int(attrs.get("shared memory", attrs.get("sharedMemory", -1))) == 58112
    record_property("route", json.dumps(dict(original=original, even=even, trace=str(path))))


@GPU
@pytest.mark.parametrize("index", (0, 1, 2))
@pytest.mark.parametrize("offset", (1, 7, 8))
def test_alignment_offsets(runtime, index, offset, record_property):
    device, native, public = runtime
    cpu = cpu_inputs((1, 65, 129, 4, 2))
    values = [x.to(device) for x in cpu]
    values[index] = offset_view(values[index], offset)
    assert values[index].data_ptr() % 16 == (2 * offset) % 16
    with torch.no_grad():
        actual, lse = native.forward(*values, .0625, True)
        public_out = public.flash_attn_func(*values, causal=True)
    assert torch.equal(actual, public_out)
    compare(actual.cpu(), lse.cpu(), cpu, .0625, record_property)


@GPU
def test_current_stream_and_positive_strided_view(runtime, record_property):
    device, native, public = runtime
    cpu = cpu_inputs((1, 65, 129, 4, 2))
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream), torch.no_grad():
        values = []
        for x in cpu:
            storage = torch.empty((*x.shape[:-1], 512), dtype=torch.float16, device=device)
            view = storage[..., ::2]
            view.copy_(x)
            values.append(view)
        out, lse = native.forward(*values, .125, True)
        public_out = public.flash_attn_func(*values, softmax_scale=.125, causal=True)
        # This read is queued on the caller stream, with no sync between input
        # construction, native attention, and dependent output consumption.
        consumed = out.clone()
        consumed_lse = lse.clone()
        public_consumed = public_out.clone()
        done = torch.cuda.Event()
        done.record()
    done.synchronize()
    assert torch.equal(consumed, public_consumed)
    compare(consumed.cpu(), consumed_lse.cpu(), cpu, .125, record_property)


@GPU
def test_known_append_causal_ln_LSE(runtime, record_property):
    device, native, _ = runtime
    q = torch.zeros((1, 3, 4, 256), device=device, dtype=torch.float16)
    k = torch.zeros((1, 7, 2, 256), device=device, dtype=torch.float16)
    v = torch.full_like(k, .5)
    with torch.no_grad():
        out, lse = native.forward(q, k, v, .0625, True)
    assert bool(torch.isfinite(out).all() and torch.isfinite(lse).all())
    assert torch.equal(out, torch.full_like(out, .5))
    expected = torch.tensor([math.log(x) for x in (5, 6, 7)], dtype=torch.float64)
    error = float((lse.cpu().double() - expected[None, None, :]).abs().max())
    record_property("known_ln_LSE_max_abs", error)
    # Calibration of log base/causal semantics, not a general corpus LSE gate.
    assert error <= 1e-5


@GPU
def test_unsupported_contracts_and_no_grad(runtime):
    device, native, public = runtime
    values = tuple(x.to(device) for x in cpu_inputs((1, 1, 33, 4, 2)))
    for scale in (0., -1., math.nan, math.inf, 1e100, 1e-100):
        with pytest.raises(RuntimeError):
            native.forward(*values, scale, True)
    with pytest.raises(RuntimeError, match="causal"):
        public.flash_attn_func(*values, causal=False)
    grad_values = tuple(x.requires_grad_() for x in values)
    with torch.enable_grad(), pytest.raises(RuntimeError, match="backward"):
        public.flash_attn_func(*grad_values, causal=True)
    with torch.no_grad():
        assert not public.flash_attn_func(*grad_values, causal=True).requires_grad
    with torch.no_grad(), pytest.raises(RuntimeError):
        native.forward(values[0].float(), values[1], values[2], .0625, True)
    with torch.no_grad(), pytest.raises(RuntimeError):
        native.forward(values[1], values[0], values[0], .0625, True)  # Lq > Lk
    packed = torch.zeros((1, 2, 3, 4, 256), device=device, dtype=torch.float16)
    with torch.no_grad(), pytest.raises(RuntimeError):
        public.flash_attn_qkvpacked_func(packed, causal=True)
    with torch.no_grad(), pytest.raises(RuntimeError):
        public.flash_attn_kvpacked_func(values[0], torch.stack(values[1:], dim=2), causal=True)
    with pytest.raises(TypeError):
        public.flash_attn_func(*values, causal=True, dropout_p=.1)
    with pytest.raises(TypeError):
        public.flash_attn_func(*values, causal=True, window_size=(1, 1))
    cu = torch.tensor([0, 1], device=device, dtype=torch.int32)
    with torch.no_grad(), pytest.raises(RuntimeError):
        public.flash_attn_varlen_func(values[0][0], values[1][0], values[2][0],
                                     cu, torch.tensor([0, 33], device=device, dtype=torch.int32),
                                     1, 33, causal=True)


def test_cpu_useful_causal_flops_and_policy():
    assert bench.useful_flops(1, 1) == 12288
    assert bench.useful_flops(2, 3) == 4 * 12 * 256 * 5


def test_cpu_ordered_ULP_signed_zero_subnormal():
    for dtype in (torch.float16, torch.float32):
        integer = torch.int16 if dtype == torch.float16 else torch.int32
        values = torch.tensor([0., -0., 1., -1.], dtype=dtype)
        subnormal = torch.tensor([1, -32767] if dtype == torch.float16 else [1, -2147483647], dtype=integer).view(dtype)
        values = torch.cat((values, subnormal))
        result = bench.error_metrics(torch, values, values.double(), dtype)
        assert result["finite"] and result["ulp_max"] == 0
        assert bench.error_metrics(torch, torch.tensor([math.nan], dtype=dtype), torch.zeros(1).double(), dtype) == {"finite": False}


def test_cpu_oracle_matches_direct_small_and_scale():
    cpu = cpu_inputs((1, 3, 7, 4, 2))
    o, lse = bench.fp64_reference(torch, cpu, [0, 1, 2], .125)
    expected_o, expected_lse = full_small_reference(cpu, .125)
    assert torch.allclose(o, expected_o, atol=1e-14, rtol=1e-14)
    assert torch.allclose(lse, expected_lse, atol=1e-14, rtol=1e-14)


def test_cpu_promoted_FI_LSE_error_rounds_only_for_ULP():
    raw_log2 = torch.tensor([1., -1., .5], dtype=torch.float32)
    promoted_ln = raw_log2.double() * math.log(2.0)
    reference = promoted_ln + 1e-10
    metric = bench.error_metrics(torch, promoted_ln, reference, torch.float32,
                                 promoted_candidate=True)
    rounded = bench.error_metrics(torch, promoted_ln.float(), reference, torch.float32)
    expected = float((promoted_ln - reference).abs().max())
    assert metric["raw_max_abs"] == expected
    assert metric["raw_max_abs"] < rounded["raw_max_abs"]
    assert metric["ulp_max"] == rounded["ulp_max"]
    assert metric["rms"] == float((promoted_ln - reference).square().mean().sqrt())
