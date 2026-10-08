# SM75 FP16 head-dimension-256 forward

The specialization supports dense append-causal inference, Q<=KV, GQA/MQA,
positive FP32-finite scales, positive-strided input views and the current
CUDA stream. Query row i attends through key KV-Q+i. The public API returns
FP16 O; the native companion also returns FP32 natural-log LSE. Misaligned
or noncontiguous inputs are materialized by the public path.

D256 backward, noncausal, packed and varlen requests are rejected. Dropout,
custom/local masks, paged caches and application integrations are outside
this change. D64/D96/D128 retain their existing behavior.

## Validation

- 108 D256 FP64 cases: O normalized RMS at most 0.0003292311. O references
  cover full outputs in 102 cases and fixed sampled rows in six; LSE is full
  in 49 cases and sampled in 59. All complete outputs were finite.
- Seven forward/backward cases per existing dimension passed on the final
  combined build; O/LSE and gradients matched the original implementation.
  Lower-D machine code was unchanged. D64/D96 steady-state performance ratios
  were 1.000084 and 0.999125. An earlier host-gap-heavy D64 run was 0.968402
  and remains a failed observation; cold-start equivalence is not claimed.
- Bounded memcheck, racecheck and synccheck passed, plus 24 alignment cases
  under memcheck. The portable suite previously ran 22 GPU and 10 CPU tests
  with no skips, covering boundaries, GQA/batches, stream dependencies,
  public/native consistency and unsupported API rejection.

FP64 is the reference; FlashInfer is a comparator. Norm-relative RMS is
`RMS(error)/max(RMS(reference),1e-12)`, not elementwise relative error.
The existing bounded O criterion is normalized RMS <= 0.001. LSE error and
nearest-rank FP16/FP32 ULP statistics are reported separately. FI log2 LSE
is promoted to FP64 before ln2 conversion for absolute/RMS analysis.

Independent comparison with the original D128 binary found comparable
engineering precision on the audited workloads. A fixed six-case supplement
used one synthetic and five captured D256 inputs, with derived first-128-
feature D128 inputs and adjusted scale; each dimension used its own FP64
reference and same-input FI comparator. Both native implementations had
O normalized RMS approximately 2.48e-4–3.09e-4 and at most 1.19% more RMS
error than their respective FI comparator. This is a bounded comparison,
not an application-quality or universal-accuracy claim.

Known differences remain: the six D256 p99.9 ULP values are 47/26/34/85/36/48
versus FI 45/24/32/81/34/46. All six fail the additional FI+1 comparator while
passing the absolute-FP64 O RMS screen. One captured case has a localized
maximum absolute error 0.002403370 versus FI 0.001502880; it is not a near-zero
ULP artifact. Historical stricter checks remain 70 pass/24 fail. These
results are disclosed unchanged; no comparator is silently redefined.
Captured tensors and internal experiment archives are not part of this PR.

## Performance

One RTX 2080 Ti; FP16 B1/Hq12/Hkv2/D256, scale 0.0625, append-causal O+LSE.
CUDA Events around CUDA Graph replay, five warmups and 30 alternating pairs.
Useful FLOPs: `4*B*Hq*D*(Q*(KV-Q)+Q*(Q+1)/2)`. No application speedup is claimed.

| Q × KV | D256 ms | FI ms | D256 useful TFLOP/s | FI/D256 |
|---|---:|---:|---:|---:|
| 4096 × 4096 | 2.177808 | 2.579856 | 47.3432 | 1.1846× |
| 8192 × 8192 | 8.124528 | 9.979808 | 50.7558 | 1.2284× |
| 8192 × 65536 | 117.163551 | 150.409744 | 52.7878 | 1.2838× |

These same-window comparisons include FI's GPU conversion to contiguous
natural-log LSE. Native scopes and every sample are also retained in the
[CSV](../benchmarks/results/d256_forward.csv), with
[environment, identities and input hashes](../benchmarks/results/d256_forward.json).
The 4096 native batch had a common timing-regime change and is not a stable
throughput headline. Separate primary/repeat native D256 medians are
52.5687/52.6430T at Q8192/KV65536 and 52.5938/52.5460T at Q8192/KV131072;
all 120 samples at those points exceed 50T.

## Reproduce

The tested environment is Linux, Python 3.12, PyTorch 2.13.0+cu130, CUDA toolkit
13.2.86, GCC 13.3 and C++17. Runtime CUDA and build toolkit versions differ.
The original README's older CUDA/PyTorch matrix and T4 have not been tested
for D256. The two extensions preserve their separate compiler policies:
D256 without fast-math, original dimensions with upstream fast-math.

```sh
git submodule update --init csrc/cutlass
CUDA_VISIBLE_DEVICES='' MAX_JOBS=2 python -m pip wheel --no-build-isolation --no-deps . -w dist
python -m pip install --no-deps dist/flash_attn_turing-*.whl
python -m pytest -q tests/test_d256_interface.py tests/test_sm75_d256_gpu.py
# Install FlashInfer 0.7.0 separately for the comparator.
python benchmarks/benchmark_sm75_d256.py --device 0 --q 8192 --kv 65536 --seed 2026092306 --scope both --out results/d256-64k
```

Run benchmarks on an idle, exclusively owned GPU. Use fresh output directories.
For the two square comparison rows, use Q=KV=4096 or 8192 and seed 2026100801.
Changing the Torch RNG/version can change inputs; compare generated input
hashes with the saved metadata. The benchmark uses FP64 on three fixed query
rows/all heads; use the small GPU tests for full-output FP64 checks. The
recorded GPU measurements preceded CLI cleanup; numerical/timing helpers
were preserved, and no GPU rerun is claimed for the cleanup.

The existing FlashAttention attribution and CUTLASS submodule/license remain.
FlashInfer is a comparison dependency; its kernel code was not copied.
The upstream snapshot has no root license or contribution-policy file.
This patch does not add a license grant; maintainers can clarify incoming
contribution terms during review.
