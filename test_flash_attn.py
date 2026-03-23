"""FlashAttention regression tests with shared helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from flash_attention_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)


# --------------------------------------------------------------------------------------
# Configuration


EXCEL_REL_EPS = 1e-6
EXCEL_TOPK_ROWS = 10_000
TEST_REL_EPS = 1e-6

SAVE_DEBUG_EXCEL = False  # flip to True to dump Excel snapshots of top errors
OUTPUT_DIR = "/outputs"

BWD_TOLS = dict(
    atol=9e-3,
    rtol=1000,
    rtol_l2=100,
    mean_atol=2e-4,
    mean_rtol=1,
    mean_rtol_l2=100,
)

DTYPES = [torch.float16]
HEAD_DIMS = [64, 128]
BATCH_SIZES = [1, 3]
SOFTMAX_SCALES = [None, 0.3]
# SOFTMAX_SCALES = [0.3]
CAUSAL_FLAGS = [False, True]
NHEAD_PAIRS = [(2, 1), (4, 2), (6, 3), (6, 1)]

SEQLEN_CASES: Sequence[Tuple[int, int]] = [
    (64, 64),
    (64, 128),
    (64, 256),
    (128, 64),
    (256, 64),
    (128, 128),
    (1024, 1024),
    (128, 256),
    (128, 1024),
    (256, 1024),
    (512, 1024),
    (256, 128),
    (512, 128),
    (768, 128),
    (1024, 128),
    (1024, 256),
    (63, 63),
    (65, 65),
    (127, 127),
    (129, 129),
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),       
    (64, 128),
    (64, 256),
    (128, 64),
    (256, 64),
    (128, 128),
    (1024, 1024),
    (128, 256),
    (128, 1024),
    (256, 1024),
    (512, 1024),
    (256, 128),
    (512, 128),
    (768, 128),
    (1024, 128),
    (1024, 256),
    (64, 2),
    (127, 63),
    (129, 65),
    (128, 127),
    (128, 129),
    (128, 1025),
    (256, 1025),
    (128, 128),
    (1024, 1024),
    (128, 256),
    (256, 64),
    (897, 1024),
    (959, 1024),
    (960, 1024),
    (961, 1024),
    (1023, 1024),
    (1024, 1023),
    (1024, 897),
    (1,64),
    (1,128),
    (65,64),
    (65,128),
    (129,64),
    (129,128),
    (257,64),
    (257,128),
    (1, 1024),
    (1023, 1024),
    (1025, 1024),
    (64, 1),
    (128,1),
    (64, 65),
    (128,65),
    (64, 129),
    (128,129),
    (64, 257),
    (128,257),
    (1024, 1),
    (1024, 2),
    (1024, 1023),
    (1024, 1025),
]


# --------------------------------------------------------------------------------------
# Helper data structures


@dataclass
class MetricsBundle:
    output: Dict[str, float]
    dq: Dict[str, float]
    dk: Dict[str, float]
    dv: Dict[str, float]

    def items(self) -> Iterable[Tuple[str, Dict[str, float]]]:
        return (("output", self.output), ("dq", self.dq), ("dk", self.dk), ("dv", self.dv))


@dataclass
class DebugPair:
    actual: torch.Tensor
    reference: torch.Tensor


@dataclass
class VarlenTensors:
    q_packed: torch.Tensor
    k_packed: torch.Tensor
    v_packed: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    d_output_packed: torch.Tensor
    q_padded: torch.Tensor
    k_padded: torch.Tensor
    v_padded: torch.Tensor
    d_output_padded: torch.Tensor
    seqlens_q: List[int]
    seqlens_k: List[int]


# --------------------------------------------------------------------------------------
# Utility functions


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FlashAttention tests")
    torch.cuda.init()
    device_index = torch.cuda.current_device()
    return torch.device(f"cuda:{device_index}")


def _tensor_with_grad(x: torch.Tensor) -> torch.Tensor:
    return x.clone().detach().requires_grad_(True)


def _error_metrics(x: torch.Tensor, ref: torch.Tensor, eps: float = TEST_REL_EPS) -> Dict[str, float]:
    diff = x - ref
    abs_err = diff.abs()
    denom = ref.abs().clamp_min(eps)
    rel_err = abs_err / denom

    diff_fp32 = diff.float()
    ref_fp32 = ref.float()
    rel_err_fp32 = rel_err.float()

    return {
        "max_abs": abs_err.max().item() if abs_err.numel() > 0 else 0.0,
        "mean_abs": abs_err.mean().item() if abs_err.numel() > 0 else 0.0,
        "max_rel": rel_err.max().item() if rel_err.numel() > 0 else 0.0,
        "mean_rel": rel_err.mean().item() if rel_err.numel() > 0 else 0.0,
        "l2_rel": (diff_fp32.norm() / (ref_fp32.norm() + eps)).item() if ref.numel() > 0 else 0.0,
        "rms_rel": rel_err_fp32.square().mean().sqrt().item() if rel_err.numel() > 0 else 0.0,
    }


def _print_metrics(bundle: MetricsBundle) -> None:
    output_metrics = bundle.output
    dq_metrics = bundle.dq
    dk_metrics = bundle.dk
    dv_metrics = bundle.dv

    print("\n")
    print("========================================")
    # print(f"seqlen_q = {seqlen_q}, seqlen_k = {seqlen_k}, d = {d}, causal = {causal}")
    print(
        f"output max_abs={output_metrics['max_abs']} mean_abs={output_metrics['mean_abs']} "
        f"max_rel={output_metrics['max_rel']} mean_rel={output_metrics['mean_rel']} "
        f"l2_rel={output_metrics['l2_rel']} rms_rel={output_metrics['rms_rel']}"
    )
    print(
        f"dQ     max_abs={dq_metrics['max_abs']} mean_abs={dq_metrics['mean_abs']} "
        f"max_rel={dq_metrics['max_rel']} mean_rel={dq_metrics['mean_rel']} "
        f"l2_rel={dq_metrics['l2_rel']} rms_rel={dq_metrics['rms_rel']}"
    )
    print(
        f"dK     max_abs={dk_metrics['max_abs']} mean_abs={dk_metrics['mean_abs']} "
        f"max_rel={dk_metrics['max_rel']} mean_rel={dk_metrics['mean_rel']} "
        f"l2_rel={dk_metrics['l2_rel']} rms_rel={dk_metrics['rms_rel']}"
    )
    print(
        f"dV     max_abs={dv_metrics['max_abs']} mean_abs={dv_metrics['mean_abs']} "
        f"max_rel={dv_metrics['max_rel']} mean_rel={dv_metrics['mean_rel']} "
        f"l2_rel={dv_metrics['l2_rel']} rms_rel={dv_metrics['rms_rel']}"
    )
    print("========================================")


def _assert_metrics(bundle: MetricsBundle) -> None:
    for name, metrics in bundle.items():
        assert metrics["max_abs"] <= BWD_TOLS["atol"], f"{name} max_abs={metrics['max_abs']}"
        assert metrics["max_rel"] <= BWD_TOLS["rtol"], f"{name} max_rel={metrics['max_rel']}"
        assert metrics["l2_rel"] <= BWD_TOLS["rtol_l2"], f"{name} l2_rel={metrics['l2_rel']}"
        assert metrics["mean_abs"] <= BWD_TOLS["mean_atol"], f"{name} mean_abs={metrics['mean_abs']}"
        assert metrics["mean_rel"] <= BWD_TOLS["mean_rtol"], f"{name} mean_rel={metrics['mean_rel']}"
        assert metrics["rms_rel"] <= BWD_TOLS["mean_rtol_l2"], f"{name} rms_rel={metrics['rms_rel']}"


def _flatten_numpy(x: torch.Tensor) -> np.ndarray:
    if x.numel() == 0:
        return np.empty(0, dtype=np.float32)
    return x.detach().float().cpu().reshape(-1).numpy()


def _build_debug_tables(pairs: Dict[str, DebugPair]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for name, tensors in pairs.items():
        actual = _flatten_numpy(tensors.actual)
        reference = _flatten_numpy(tensors.reference)
        diff = actual - reference
        abs_diff = np.abs(diff)
        rel_diff = abs_diff / np.maximum(np.abs(reference), EXCEL_REL_EPS)
        base_df = pd.DataFrame(
            {
                "actual": actual,
                "reference": reference,
                "diff": diff,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
            }
        )
        tables[f"{name}_abs"] = base_df.sort_values(by="abs_diff", ascending=False).head(EXCEL_TOPK_ROWS)
        tables[f"{name}_rel"] = base_df.sort_values(by="rel_diff", ascending=False).head(EXCEL_TOPK_ROWS)
    return tables


def _maybe_emit_excel(tag: str, pairs: Dict[str, DebugPair]) -> None:
    if not SAVE_DEBUG_EXCEL:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"{timestamp}_{tag}.xlsx")
    tables = _build_debug_tables(pairs)

    with pd.ExcelWriter(path) as writer:
        for sheet_name, df in tables.items():
            writer_sheet = sheet_name[:31]
            df.to_excel(writer, sheet_name=writer_sheet, index=False)

    print(f"Saved Excel debug file: {path}")


def causal_lower_right(seqlen_q: int, seqlen_k: int, device: torch.device) -> torch.Tensor:
    diagonal_offset = seqlen_k - seqlen_q
    return torch.tril(
        torch.ones((seqlen_q, seqlen_k), dtype=torch.bool, device=device),
        diagonal=diagonal_offset,
    )


def vanilla_attention_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    d_output: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, ...]:
    query_torch = query.permute(0, 2, 1, 3).contiguous().clone().requires_grad_(True)
    key_torch = key.permute(0, 2, 1, 3).contiguous().clone().requires_grad_(True)
    value_torch = value.permute(0, 2, 1, 3).contiguous().clone().requires_grad_(True)

    nheads_q = query_torch.size(1)
    nheads_k = key_torch.size(1)
    assert nheads_q % nheads_k == 0, "nheads_q must be divisible by nheads_k"
    enable_gqa = nheads_q != nheads_k

    seqlen_q = query_torch.size(2)
    seqlen_k = key_torch.size(2)

    is_causal = False
    attn_mask = None
    if causal:
        if seqlen_q == seqlen_k:
            is_causal = True
        else:
            attn_mask = causal_lower_right(seqlen_q, seqlen_k, device=query_torch.device)

    output_torch = F.scaled_dot_product_attention(
        query_torch,
        key_torch,
        value_torch,
        attn_mask=attn_mask,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
        scale=softmax_scale,
    )

    if d_output is None:
        return (output_torch.permute(0, 2, 1, 3).contiguous(),)

    d_output_torch = d_output.permute(0, 2, 1, 3).contiguous()
    d_query_torch, d_key_torch, d_value_torch = torch.autograd.grad(
        outputs=output_torch,
        inputs=(query_torch, key_torch, value_torch),
        grad_outputs=d_output_torch,
        retain_graph=False,
        allow_unused=False,
    )

    return (
        output_torch.permute(0, 2, 1, 3).contiguous(),
        d_query_torch.permute(0, 2, 1, 3).contiguous(),
        d_key_torch.permute(0, 2, 1, 3).contiguous(),
        d_value_torch.permute(0, 2, 1, 3).contiguous(),
    )


def memory_efficient_attention_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    d_output: torch.Tensor,
    causal: bool,
    softmax_scale: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return vanilla_attention_ref(query, key, value, d_output, causal=causal, softmax_scale=softmax_scale)


def _pack_padded_tensor(x: torch.Tensor, seqlens: Sequence[int]) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for i, seqlen in enumerate(seqlens):
        if seqlen <= 0:
            continue
        chunks.append(x[i, :seqlen].contiguous())
    if not chunks:
        return x.new_zeros((0,) + x.shape[2:])
    return torch.cat(chunks, dim=0)


def _generate_varlen_tensors(
    *,
    batch_size: int,
    nheads: int,
    nheads_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> VarlenTensors:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive for varlen tests")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)

    seqlens_q = torch.randint(1, max_seqlen_q + 1, (batch_size,), dtype=torch.int32, generator=rng)
    seqlens_k = torch.randint(1, max_seqlen_k + 1, (batch_size,), dtype=torch.int32, generator=rng)
    seqlens_q[torch.randint(0, batch_size, (1,), generator=rng).item()] = max_seqlen_q
    seqlens_k[torch.randint(0, batch_size, (1,), generator=rng).item()] = max_seqlen_k

    cu_q = torch.zeros(batch_size + 1, dtype=torch.int32)
    cu_k = torch.zeros(batch_size + 1, dtype=torch.int32)
    cu_q[1:] = torch.cumsum(seqlens_q, dim=0, dtype=torch.int32)
    cu_k[1:] = torch.cumsum(seqlens_k, dim=0, dtype=torch.int32)

    total_q = int(cu_q[-1].item())
    total_k = int(cu_k[-1].item())

    q_packed = torch.randn(total_q, nheads, head_dim, device=device, dtype=dtype)
    k_packed = torch.randn(total_k, nheads_k, head_dim, device=device, dtype=dtype)
    v_packed = torch.randn(total_k, nheads_k, head_dim, device=device, dtype=dtype)

    q_padded = torch.zeros(batch_size, max_seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    k_padded = torch.zeros(batch_size, max_seqlen_k, nheads_k, head_dim, device=device, dtype=dtype)
    v_padded = torch.zeros(batch_size, max_seqlen_k, nheads_k, head_dim, device=device, dtype=dtype)

    q_offset = 0
    for i, seqlen in enumerate(seqlens_q.tolist()):
        next_offset = q_offset + seqlen
        q_padded[i, :seqlen] = q_packed[q_offset:next_offset]
        q_offset = next_offset

    k_offset = 0
    for i, seqlen in enumerate(seqlens_k.tolist()):
        next_offset = k_offset + seqlen
        k_slice = k_packed[k_offset:next_offset]
        v_slice = v_packed[k_offset:next_offset]
        k_padded[i, :seqlen] = k_slice
        v_padded[i, :seqlen] = v_slice
        k_offset = next_offset

    d_output_padded = torch.randn(batch_size, max_seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    d_output_packed = _pack_padded_tensor(d_output_padded, [int(x) for x in seqlens_q.tolist()])

    return VarlenTensors(
        q_packed=q_packed,
        k_packed=k_packed,
        v_packed=v_packed,
        cu_seqlens_q=cu_q.to(device=device),
        cu_seqlens_k=cu_k.to(device=device),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        d_output_packed=d_output_packed,
        q_padded=q_padded,
        k_padded=k_padded,
        v_padded=v_padded,
        d_output_padded=d_output_padded,
        seqlens_q=[int(x) for x in seqlens_q.tolist()],
        seqlens_k=[int(x) for x in seqlens_k.tolist()],
    )


def _varlen_reference(
    tensors: VarlenTensors,
    *,
    causal: bool,
    softmax_scale: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    output_ref: List[torch.Tensor] = []
    dq_ref: List[torch.Tensor] = []
    dk_ref: List[torch.Tensor] = []
    dv_ref: List[torch.Tensor] = []

    for i, (seqlen_q, seqlen_k) in enumerate(zip(tensors.seqlens_q, tensors.seqlens_k)):
        q_i = tensors.q_padded[i : i + 1, :seqlen_q]
        k_i = tensors.k_padded[i : i + 1, :seqlen_k]
        v_i = tensors.v_padded[i : i + 1, :seqlen_k]
        d_out_i = tensors.d_output_padded[i : i + 1, :seqlen_q]

        out_i, dq_i, dk_i, dv_i = vanilla_attention_ref(
            q_i,
            k_i,
            v_i,
            d_out_i,
            causal=causal,
            softmax_scale=softmax_scale,
        )

        output_ref.append(out_i.squeeze(0))
        dq_ref.append(dq_i.squeeze(0))
        dk_ref.append(dk_i.squeeze(0))
        dv_ref.append(dv_i.squeeze(0))

    return (
        torch.cat(output_ref, dim=0),
        torch.cat(dq_ref, dim=0),
        torch.cat(dk_ref, dim=0),
        torch.cat(dv_ref, dim=0),
    )


def _bundle_from_tensors(
    output: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    output_ref: torch.Tensor,
    dq_ref: torch.Tensor,
    dk_ref: torch.Tensor,
    dv_ref: torch.Tensor,
) -> Tuple[MetricsBundle, Dict[str, DebugPair]]:
    bundle = MetricsBundle(
        output=_error_metrics(output, output_ref),
        dq=_error_metrics(dq, dq_ref),
        dk=_error_metrics(dk, dk_ref),
        dv=_error_metrics(dv, dv_ref),
    )
    pairs = {
        "output": DebugPair(output, output_ref),
        "dq": DebugPair(dq, dq_ref),
        "dk": DebugPair(dk, dk_ref),
        "dv": DebugPair(dv, dv_ref),
    }
    return bundle, pairs


# --------------------------------------------------------------------------------------
# Regular attention tests


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("nheads, nheads_k", NHEAD_PAIRS)
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
@pytest.mark.parametrize("softmax_scale", SOFTMAX_SCALES)
@pytest.mark.parametrize("seqlen_q, seqlen_k", SEQLEN_CASES)
def test_flash_attn(
    batch_size: int,
    nheads: int,
    nheads_k: int,
    seqlen_q: int,
    seqlen_k: int,
    head_dim: int,
    softmax_scale: Optional[float],
    causal: bool,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()

    query = torch.randn(batch_size, seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, seqlen_k, nheads_k, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, seqlen_k, nheads_k, head_dim, device=device, dtype=dtype)
    d_output = torch.randn(batch_size, seqlen_q, nheads, head_dim, device=device, dtype=dtype)

    q_flash = _tensor_with_grad(query)
    k_flash = _tensor_with_grad(key)
    v_flash = _tensor_with_grad(value)

    output_flash = flash_attn_func(
        q_flash,
        k_flash,
        v_flash,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    torch.cuda.synchronize()

    dq_flash, dk_flash, dv_flash = torch.autograd.grad(
        outputs=output_flash,
        inputs=(q_flash, k_flash, v_flash),
        grad_outputs=d_output.contiguous(),
        retain_graph=False,
        allow_unused=False,
    )
    torch.cuda.synchronize()

    output_ref, dq_ref, dk_ref, dv_ref = memory_efficient_attention_ref(
        query,
        key,
        value,
        d_output,
        causal,
        softmax_scale,
    )

    bundle, pairs = _bundle_from_tensors(
        output_flash.detach(),
        dq_flash.detach(),
        dk_flash.detach(),
        dv_flash.detach(),
        output_ref.detach(),
        dq_ref.detach(),
        dk_ref.detach(),
        dv_ref.detach(),
    )

    _print_metrics(bundle)
    _maybe_emit_excel(
        tag=f"flash_attn_b{batch_size}_hq{nheads}_hk{nheads_k}_sq{seqlen_q}_sk{seqlen_k}_d{head_dim}_c{int(causal)}",
        pairs=pairs,
    )
    _assert_metrics(bundle)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("nheads, nheads_k", NHEAD_PAIRS)
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
@pytest.mark.parametrize("softmax_scale", SOFTMAX_SCALES)
@pytest.mark.parametrize("seqlen_q, seqlen_k", SEQLEN_CASES)
def test_flash_attn_kv(
    batch_size: int,
    nheads: int,
    nheads_k: int,
    seqlen_q: int,
    seqlen_k: int,
    head_dim: int,
    softmax_scale: Optional[float],
    causal: bool,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()

    query = torch.randn(batch_size, seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, seqlen_k, nheads_k, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, seqlen_k, nheads_k, head_dim, device=device, dtype=dtype)
    d_output = torch.randn(batch_size, seqlen_q, nheads, head_dim, device=device, dtype=dtype)

    query_flash = _tensor_with_grad(query)
    kv_flash = _tensor_with_grad(torch.stack((key, value), dim=2))

    output_flash = flash_attn_kvpacked_func(
        query_flash,
        kv_flash,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    torch.cuda.synchronize()

    dq_flash, dkv_flash = torch.autograd.grad(
        outputs=output_flash,
        inputs=(query_flash, kv_flash),
        grad_outputs=d_output.contiguous(),
        retain_graph=False,
        allow_unused=False,
    )
    torch.cuda.synchronize()

    dk_flash = dkv_flash[:, :, 0].detach()
    dv_flash = dkv_flash[:, :, 1].detach()

    output_ref, dq_ref, dk_ref, dv_ref = memory_efficient_attention_ref(
        query,
        key,
        value,
        d_output,
        causal,
        softmax_scale,
    )

    bundle, pairs = _bundle_from_tensors(
        output_flash.detach(),
        dq_flash.detach(),
        dk_flash,
        dv_flash,
        output_ref.detach(),
        dq_ref.detach(),
        dk_ref.detach(),
        dv_ref.detach(),
    )

    _print_metrics(bundle)
    _maybe_emit_excel(
        tag=f"flash_attn_kv_b{batch_size}_hq{nheads}_hk{nheads_k}_sq{seqlen_q}_sk{seqlen_k}_d{head_dim}_c{int(causal)}",
        pairs=pairs,
    )
    _assert_metrics(bundle)


EQUAL_SEQLEN_CASES = [case for case in SEQLEN_CASES if case[0] == case[1]]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("nheads, nheads_k", NHEAD_PAIRS)
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
@pytest.mark.parametrize("softmax_scale", SOFTMAX_SCALES)
@pytest.mark.parametrize("seqlen", sorted({case[0] for case in EQUAL_SEQLEN_CASES}))
def test_flash_attn_qkv(
    batch_size: int,
    nheads: int,
    nheads_k: int,
    seqlen: int,
    head_dim: int,
    softmax_scale: Optional[float],
    causal: bool,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()

    query = torch.randn(batch_size, seqlen, nheads, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, seqlen, nheads_k, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, seqlen, nheads_k, head_dim, device=device, dtype=dtype)
    d_output = torch.randn(batch_size, seqlen, nheads, head_dim, device=device, dtype=dtype)

    qkv_flash = _tensor_with_grad(torch.stack((query, key, value), dim=2))

    output_flash = flash_attn_qkvpacked_func(
        qkv_flash,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    torch.cuda.synchronize()

    (dqkv_flash,) = torch.autograd.grad(
        outputs=output_flash,
        inputs=(qkv_flash,),
        grad_outputs=d_output.contiguous(),
        retain_graph=False,
        allow_unused=False,
    )
    torch.cuda.synchronize()

    dq_flash = dqkv_flash[:, :, 0].detach()
    dk_flash = dqkv_flash[:, :, 1].detach()
    dv_flash = dqkv_flash[:, :, 2].detach()

    output_ref, dq_ref, dk_ref, dv_ref = memory_efficient_attention_ref(
        query,
        key,
        value,
        d_output,
        causal,
        softmax_scale,
    )

    bundle, pairs = _bundle_from_tensors(
        output_flash.detach(),
        dq_flash,
        dk_flash,
        dv_flash,
        output_ref.detach(),
        dq_ref.detach(),
        dk_ref.detach(),
        dv_ref.detach(),
    )

    _print_metrics(bundle)
    _maybe_emit_excel(
        tag=f"flash_attn_qkv_b{batch_size}_hq{nheads}_hk{nheads_k}_s{seqlen}_d{head_dim}_c{int(causal)}",
        pairs=pairs,
    )
    _assert_metrics(bundle)


# --------------------------------------------------------------------------------------
# Variable-length attention tests


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("nheads, nheads_k", NHEAD_PAIRS)
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
@pytest.mark.parametrize("softmax_scale", SOFTMAX_SCALES)
@pytest.mark.parametrize("max_seqlen_q, max_seqlen_k", SEQLEN_CASES)
def test_flash_attn_varlen(
    batch_size: int,
    nheads: int,
    nheads_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    head_dim: int,
    softmax_scale: Optional[float],
    causal: bool,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()
    tensors = _generate_varlen_tensors(
        batch_size=batch_size,
        nheads=nheads,
        nheads_k=nheads_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    q_flash = _tensor_with_grad(tensors.q_packed)
    k_flash = _tensor_with_grad(tensors.k_packed)
    v_flash = _tensor_with_grad(tensors.v_packed)

    out_flash = flash_attn_varlen_func(
        q_flash,
        k_flash,
        v_flash,
        tensors.cu_seqlens_q,
        tensors.cu_seqlens_k,
        tensors.max_seqlen_q,
        tensors.max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    torch.cuda.synchronize()

    dq_flash, dk_flash, dv_flash = torch.autograd.grad(
        outputs=out_flash,
        inputs=(q_flash, k_flash, v_flash),
        grad_outputs=tensors.d_output_packed.contiguous(),
        retain_graph=False,
        allow_unused=False,
    )
    torch.cuda.synchronize()

    output_ref, dq_ref, dk_ref, dv_ref = _varlen_reference(
        tensors,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    bundle, pairs = _bundle_from_tensors(
        out_flash.detach(),
        dq_flash.detach(),
        dk_flash.detach(),
        dv_flash.detach(),
        output_ref.detach(),
        dq_ref.detach(),
        dk_ref.detach(),
        dv_ref.detach(),
    )

    _print_metrics(bundle)
    _maybe_emit_excel(
        tag=(
            f"flash_attn_varlen_b{batch_size}_hq{nheads}_hk{nheads_k}_"
            f"mq{max_seqlen_q}_mk{max_seqlen_k}_d{head_dim}_c{int(causal)}"
        ),
        pairs=pairs,
    )
    _assert_metrics(bundle)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("nheads, nheads_k", NHEAD_PAIRS)
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
@pytest.mark.parametrize("softmax_scale", SOFTMAX_SCALES)
@pytest.mark.parametrize("max_seqlen_q, max_seqlen_k", SEQLEN_CASES)
def test_flash_attn_varlen_kv(
    batch_size: int,
    nheads: int,
    nheads_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    head_dim: int,
    softmax_scale: Optional[float],
    causal: bool,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()
    tensors = _generate_varlen_tensors(
        batch_size=batch_size,
        nheads=nheads,
        nheads_k=nheads_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    q_flash = _tensor_with_grad(tensors.q_packed)
    kv_flash = _tensor_with_grad(torch.stack((tensors.k_packed, tensors.v_packed), dim=1))

    out_flash = flash_attn_varlen_kvpacked_func(
        q_flash,
        kv_flash,
        tensors.cu_seqlens_q,
        tensors.cu_seqlens_k,
        tensors.max_seqlen_q,
        tensors.max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    torch.cuda.synchronize()

    dq_flash, dkv_flash = torch.autograd.grad(
        outputs=out_flash,
        inputs=(q_flash, kv_flash),
        grad_outputs=tensors.d_output_packed.contiguous(),
        retain_graph=False,
        allow_unused=False,
    )
    torch.cuda.synchronize()

    dk_flash = dkv_flash[:, 0].detach()
    dv_flash = dkv_flash[:, 1].detach()

    output_ref, dq_ref, dk_ref, dv_ref = _varlen_reference(
        tensors,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    bundle, pairs = _bundle_from_tensors(
        out_flash.detach(),
        dq_flash.detach(),
        dk_flash,
        dv_flash,
        output_ref.detach(),
        dq_ref.detach(),
        dk_ref.detach(),
        dv_ref.detach(),
    )

    _print_metrics(bundle)
    _maybe_emit_excel(
        tag=(
            f"flash_attn_varlen_kv_b{batch_size}_hq{nheads}_hk{nheads_k}_"
            f"mq{max_seqlen_q}_mk{max_seqlen_k}_d{head_dim}_c{int(causal)}"
        ),
        pairs=pairs,
    )
    _assert_metrics(bundle)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("nheads, nheads_k", NHEAD_PAIRS)
@pytest.mark.parametrize("causal", CAUSAL_FLAGS)
@pytest.mark.parametrize("softmax_scale", SOFTMAX_SCALES)
@pytest.mark.parametrize("max_seqlen", sorted({case[0] for case in SEQLEN_CASES if case[0] == case[1]}))
def test_flash_attn_varlen_qkv(
    batch_size: int,
    nheads: int,
    nheads_k: int,
    max_seqlen: int,
    head_dim: int,
    softmax_scale: Optional[float],
    causal: bool,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()

    tensors = _generate_varlen_tensors(
        batch_size=batch_size,
        nheads=nheads,
        nheads_k=nheads_k,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )

    qkv_packed = torch.stack((tensors.q_packed, tensors.k_packed, tensors.v_packed), dim=1)
    qkv_flash = _tensor_with_grad(qkv_packed)

    out_flash = flash_attn_varlen_qkvpacked_func(
        qkv_flash,
        tensors.cu_seqlens_q,
        tensors.max_seqlen_q,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    torch.cuda.synchronize()

    (dqkv_flash,) = torch.autograd.grad(
        outputs=out_flash,
        inputs=(qkv_flash,),
        grad_outputs=tensors.d_output_packed.contiguous(),
        retain_graph=False,
        allow_unused=False,
    )
    torch.cuda.synchronize()

    dq_flash = dqkv_flash[:, 0].detach()
    dk_flash = dqkv_flash[:, 1].detach()
    dv_flash = dqkv_flash[:, 2].detach()

    output_ref, dq_ref, dk_ref, dv_ref = _varlen_reference(
        tensors,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    bundle, pairs = _bundle_from_tensors(
        out_flash.detach(),
        dq_flash,
        dk_flash,
        dv_flash,
        output_ref.detach(),
        dq_ref.detach(),
        dk_ref.detach(),
        dv_ref.detach(),
    )

    _print_metrics(bundle)
    _maybe_emit_excel(
        tag=(
            f"flash_attn_varlen_qkv_b{batch_size}_hq{nheads}_hk{nheads_k}_"
            f"m{max_seqlen}_d{head_dim}_c{int(causal)}"
        ),
        pairs=pairs,
    )
    _assert_metrics(bundle)
