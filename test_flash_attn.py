
import pytest
import torch
import os
from datetime import datetime

import torch.nn.functional as F
from torch.nn.attention import bias
import pandas as pd
import numpy as np

from flash_attention_interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

torch.set_printoptions(threshold=torch.inf)

EXCEL_TOPK_ROWS = 10_000
EXCEL_REL_EPS = 1e-6
TEST_REL_EPS = 1e-6
SAVE_FAIL_DEBUG_EXCEL = True

BWD_TOLS = dict(
    atol=5e-3,
    rtol=1000,
    rtol_l2=100,
    mean_atol=2e-4,
    mean_rtol=1, #1e-2 
    mean_rtol_l2=100,
)


def _topk_rows_by_score(df, score, k=EXCEL_TOPK_ROWS):
    sorted_idx = score.sort_values(ascending=False).index
    if len(df) <= k:
        return df.reindex(sorted_idx)
    return df.reindex(sorted_idx[:k])


def _max_abs_score(df, cols):
    return df[cols].abs().max(axis=1)


def _max_rel_score(df, diff_ref_pairs, eps=EXCEL_REL_EPS):
    rel_scores = []
    for diff_col, ref_col in diff_ref_pairs:
        denom = df[ref_col].abs().clip(lower=eps)
        rel_scores.append(df[diff_col].abs() / denom)
    return pd.concat(rel_scores, axis=1).max(axis=1)


def _rel_err_col(df, diff_col, ref_col, eps=EXCEL_REL_EPS):
    denom = df[ref_col].abs().clip(lower=eps)
    return df[diff_col].abs() / denom


def _error_metrics(x, ref, eps=TEST_REL_EPS):
    diff = x - ref
    abs_err = diff.abs()
    denom = ref.abs().clamp_min(eps)
    rel_err = abs_err / denom

    diff_fp32 = diff.float()
    ref_fp32 = ref.float()
    rel_err_fp32 = rel_err.float()

    return {
        "max_abs": abs_err.max().item(),
        "mean_abs": abs_err.mean().item(),
        "max_rel": rel_err.max().item(),
        "mean_rel": rel_err.mean().item(),
        "l2_rel": (
            diff_fp32.norm() / (ref_fp32.norm() + eps)
        ).item(),
        # RMS of elementwise relative error: "average" counterpart to max_rel / l2-style check.
        "rms_rel": rel_err_fp32.square().mean().sqrt().item(),
    }


def _make_identity_sequence(num_rows, num_heads, head_dim, device, dtype):
    tensor = torch.zeros(num_rows, num_heads, head_dim, device=device, dtype=dtype)
    if num_rows == 0:
        return tensor
    cols = torch.arange(num_rows, device=device) % head_dim
    cols = cols.view(-1, 1, 1).expand(-1, num_heads, 1)
    ones = torch.ones(num_rows, num_heads, 1, device=device, dtype=dtype)
    tensor.scatter_(2, cols, ones)
    return tensor


def _fill_identity_padded_(dest, seqlens_q):
    if dest.numel() == 0:
        return
    batch_size = dest.size(0)
    nheads = dest.size(2)
    head_dim = dest.size(3)
    device = dest.device
    dtype = dest.dtype
    for b in range(batch_size):
        seqlen = int(seqlens_q[b].item())
        if seqlen <= 0:
            continue
        identity_block = _make_identity_sequence(seqlen, nheads, head_dim, device, dtype)
        dest[b, :seqlen].copy_(identity_block)


def _make_identity_packed(seqlens, num_heads, head_dim, device, dtype):
    chunks = []
    for seqlen in seqlens.tolist():
        if seqlen <= 0:
            continue
        chunks.append(_make_identity_sequence(seqlen, num_heads, head_dim, device, dtype))
    if chunks:
        return torch.cat(chunks, dim=0)
    return torch.zeros((0, num_heads, head_dim), device=device, dtype=dtype)


def _assert_metrics(metrics, atol, rtol, rtol_l2, mean_atol, mean_rtol, mean_rtol_l2, name):
    assert metrics["max_abs"] <= atol, f"{name} max_abs={metrics['max_abs']} > atol={atol}"
    assert metrics["max_rel"] <= rtol, f"{name} max_rel={metrics['max_rel']} > rtol={rtol}"
    assert metrics["l2_rel"] <= rtol_l2, f"{name} l2_rel={metrics['l2_rel']} > rtol_l2={rtol_l2}"
    assert metrics["mean_abs"] <= mean_atol, f"{name} mean_abs={metrics['mean_abs']} > mean_atol={mean_atol}"
    assert metrics["mean_rel"] <= mean_rtol, f"{name} mean_rel={metrics['mean_rel']} > mean_rtol={mean_rtol}"
    assert metrics["rms_rel"] <= mean_rtol_l2, f"{name} rms_rel={metrics['rms_rel']} > mean_rtol_l2={mean_rtol_l2}"


def causal_lower_right(seqlen_q, seqlen_k, device):
    """
    Build a lower-right causal mask for seqlen_q >= seqlen_k.
    Returns a boolean tensor of shape (seqlen_q, seqlen_k)
    """
    diagonal_offset = seqlen_k - seqlen_q
    mask = torch.tril(
        torch.ones((seqlen_q, seqlen_k), dtype=torch.bool, device=device),
        diagonal=diagonal_offset,
    )
    return mask  # bool tensor (seqlen_q, seqlen_k)


def vanilla_attention_ref(query, key, value, d_output=None, causal=False):
    """
    Reference attention implementation for testing FlashAttention.

    Args:
        query, key, value: (batch, seqlen, nheads, d) tensors
        d_output: gradient of output for backward (same shape)
        causal: whether to apply causal mask

    Returns:
        If d_output is None:
            output: (batch, seqlen, nheads, d)
        Else:
            output, d_query, d_key, d_value
    """
    # make leaf tensors that require grad
    query_torch = query.permute(0, 2, 1, 3).contiguous().clone().requires_grad_(True)
    key_torch   = key.permute(0, 2, 1, 3).contiguous().clone().requires_grad_(True)
    value_torch = value.permute(0, 2, 1, 3).contiguous().clone().requires_grad_(True)

    batch_size, nheads_q, seqlen_q, d = query_torch.size()
    _, nheads_k, seqlen_k, _ = key_torch.size()
    assert nheads_q % nheads_k == 0, "nheads_q must be divisible by nheads_k for GQA/MQA"
    nheads_ratio = nheads_q // nheads_k
    key_attn = key_torch.repeat_interleave(nheads_ratio, dim=1) if nheads_q != nheads_k else key_torch
    value_attn = value_torch.repeat_interleave(nheads_ratio, dim=1) if nheads_q != nheads_k else value_torch

    # compute attention scores
    scores = torch.matmul(query_torch, key_attn.transpose(-2, -1)) / (d ** 0.5)

    if causal:
        attn_mask = causal_lower_right(seqlen_q, seqlen_k, device=scores.device)
        attn_mask = attn_mask.view(1, 1, seqlen_q, seqlen_k)  # broadcast
        scores = scores.masked_fill(~attn_mask, float("-inf"))

    # softmax
    attn = F.softmax(scores, dim=-1)
    attn = torch.where(attn.isnan(), torch.zeros_like(attn), attn)

    # compute output
    output_torch = torch.matmul(attn, value_attn)  # (batch, nheads_q, seqlen_q, d)
    # print(f"inside 1, output_torch = {output_torch}")
    if d_output is None:
        # forward only
        return output_torch.permute(0, 2, 1, 3).contiguous()
    else:
        # backward pass
        d_output_torch = d_output.permute(0, 2, 1, 3).contiguous()
        d_query_torch, d_key_torch, d_value_torch = torch.autograd.grad(
            outputs=output_torch,
            inputs=(query_torch, key_torch, value_torch),
            grad_outputs=d_output_torch,
            retain_graph=False,
            allow_unused=False  # ensure no None
        )

        # permute back to original shape
        return (
            output_torch.permute(0, 2, 1, 3).contiguous(),
            d_query_torch.permute(0, 2, 1, 3).contiguous(),
            d_key_torch.permute(0, 2, 1, 3).contiguous(),
            d_value_torch.permute(0, 2, 1, 3).contiguous(),
        )


# this is for comparing with pytorch sdpa with vanilla attention
def memory_efficient_attention_ref(query, key, value, d_output=None, causal=False):
    query_torch = query.permute(0, 2, 1, 3).contiguous().clone().requires_grad_(True)
    key_torch   = key.permute(0, 2, 1, 3).contiguous().clone().requires_grad_(True)
    value_torch = value.permute(0, 2, 1, 3).contiguous().clone().requires_grad_(True)

    nheads_q = query_torch.size(1)
    nheads_k = key_torch.size(1)
    assert nheads_q % nheads_k == 0, "nheads_q must be divisible by nheads_k for GQA/MQA"
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
        query=query_torch,
        key=key_torch,
        value=value_torch,
        attn_mask=attn_mask,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
    )

    if d_output is None:
        return output_torch.permute(0, 2, 1, 3).contiguous()
    else:
        d_output_torch = d_output.permute(0, 2, 1, 3).contiguous()
        d_query_torch, d_key_torch, d_value_torch = torch.autograd.grad(
            outputs=output_torch,
            inputs=(query_torch, key_torch, value_torch),
            grad_outputs=d_output_torch,
            retain_graph=False,
            allow_unused=False  # ensure no None
        )

        # permute back to original shape
        return (
            output_torch.permute(0, 2, 1, 3).contiguous(),
            d_query_torch.permute(0, 2, 1, 3).contiguous(),
            d_key_torch.permute(0, 2, 1, 3).contiguous(),
            d_value_torch.permute(0, 2, 1, 3).contiguous(),
        )


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("d", [64, 128])
# @pytest.mark.parametrize("d", [64])
# @pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("batch_size", [1, 3])
# @pytest.mark.parametrize("nheads, nheads_k", [(16, 16), (16, 4), (16, 1)])
@pytest.mark.parametrize("nheads, nheads_k", [(2, 1), (4, 2), (6, 3), (6, 1)])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [True])
# @pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize(
    "seqlen_q, seqlen_k",
    [
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
        (1024, 1023),
        (1024, 1025),
    ],
)
def test_flash_attn_bwd(        
    batch_size, nheads, nheads_k, seqlen_q, seqlen_k,  d, causal, dtype
):
    
    torch.cuda.init()
    torch.cuda.current_device()
    device = "cuda"    
    query = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    key = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    value = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    d_output = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)

    query_flash = query.clone().detach().requires_grad_(True)
    key_flash = key.clone().detach().requires_grad_(True)
    value_flash = value.clone().detach().requires_grad_(True)

    torch.set_printoptions(
        threshold=1_000_000,  # big enough to avoid truncation
        linewidth=300,
        precision=1,
        sci_mode=False
    )


    # output_torch,  d_query_torch, d_key_torch, d_value_torch = vanilla_attention_ref(query, key, value, d_output, causal)

    output_torch,  d_query_torch, d_key_torch, d_value_torch = memory_efficient_attention_ref(query, key, value, d_output, causal)


    torch.cuda.synchronize()


    output_torch = output_torch.detach().clone()
    d_query_torch = d_query_torch.detach().clone()
    d_key_torch = d_key_torch.detach().clone()
    d_value_torch = d_value_torch.detach().clone()

    output_flash = flash_attn_func(
        query_flash,
        key_flash,
        value_flash,
        causal=causal,
    )
    torch.cuda.synchronize()
    grad_output = d_output.contiguous()
    d_query, d_key, d_value = torch.autograd.grad(
        outputs=output_flash,
        inputs=(query_flash, key_flash, value_flash),
        grad_outputs=grad_output,
        retain_graph=False,
        allow_unused=False,
    )
    torch.cuda.synchronize()

    output = output_flash.detach()
    d_query = d_query.detach()
    d_key = d_key.detach()
    d_value = d_value.detach()

    output_metrics = _error_metrics(output, output_torch)
    dq_metrics = _error_metrics(d_query, d_query_torch)
    dk_metrics = _error_metrics(d_key, d_key_torch)
    dv_metrics = _error_metrics(d_value, d_value_torch)


    print("\n")
    print("========================================")
    # print(f"seqlen_q = {seqlen_q}, seqlen_k = {seqlen_k}, d = {d}, causal = {causal}")
    print(f"output max_abs={output_metrics['max_abs']} mean_abs={output_metrics['mean_abs']} "
          f"max_rel={output_metrics['max_rel']} mean_rel={output_metrics['mean_rel']} "
          f"l2_rel={output_metrics['l2_rel']} rms_rel={output_metrics['rms_rel']}")
    print(f"dQ     max_abs={dq_metrics['max_abs']} mean_abs={dq_metrics['mean_abs']} "
          f"max_rel={dq_metrics['max_rel']} mean_rel={dq_metrics['mean_rel']} "
          f"l2_rel={dq_metrics['l2_rel']} rms_rel={dq_metrics['rms_rel']}")
    print(f"dK     max_abs={dk_metrics['max_abs']} mean_abs={dk_metrics['mean_abs']} "
          f"max_rel={dk_metrics['max_rel']} mean_rel={dk_metrics['mean_rel']} "
          f"l2_rel={dk_metrics['l2_rel']} rms_rel={dk_metrics['rms_rel']}")
    print(f"dV     max_abs={dv_metrics['max_abs']} mean_abs={dv_metrics['mean_abs']} "
          f"max_rel={dv_metrics['max_rel']} mean_rel={dv_metrics['mean_rel']} "
          f"l2_rel={dv_metrics['l2_rel']} rms_rel={dv_metrics['rms_rel']}")
    print("========================================")


    bwd_tols = BWD_TOLS
    failed = any(
        not (
            m["max_abs"] <= bwd_tols["atol"] and
            m["max_rel"] <= bwd_tols["rtol"] and
            m["l2_rel"] <= bwd_tols["rtol_l2"] and
            m["mean_abs"] <= bwd_tols["mean_atol"] and
            m["mean_rel"] <= bwd_tols["mean_rtol"] and
            m["rms_rel"] <= bwd_tols["mean_rtol_l2"]
        )
        for m in (output_metrics, dq_metrics, dk_metrics, dv_metrics)
    )
    if failed and SAVE_FAIL_DEBUG_EXCEL:


        query_np = query.detach().cpu().numpy()
        key_np = key.detach().cpu().numpy()
        value_np = value.detach().cpu().numpy()

        output_diff_np = (output - output_torch).detach().cpu().numpy()
        output_np = output.detach().cpu().numpy()
        output_torch_np = output_torch.detach().cpu().numpy()

        d_query_diff_np = (d_query - d_query_torch).detach().cpu().numpy()
        d_query_np = d_query.detach().cpu().numpy()
        d_query_torch_np = d_query_torch.detach().cpu().numpy()

        d_key_diff_np = (d_key - d_key_torch).detach().cpu().numpy()
        d_key_np = d_key.detach().cpu().numpy()
        d_key_torch_np = d_key_torch.detach().cpu().numpy()

        d_value_diff_np = (d_value - d_value_torch).detach().cpu().numpy()
        d_value_np = d_value.detach().cpu().numpy()
        d_value_torch_np = d_value_torch.detach().cpu().numpy()

        b_idx, s_idx, h_idx, d_idx = np.indices(output_np.shape)
        df = pd.DataFrame({
            "batch": b_idx.flatten(),
            "seqlen": s_idx.flatten(),
            "head": h_idx.flatten(),
            "d": d_idx.flatten(),
            "output_diff": output_diff_np.flatten(),
            "output": output_np.flatten(),
            "output_torch": output_torch_np.flatten(),
            "d_query_diff": d_query_diff_np.flatten(),
            "d_query": d_query_np.flatten(),
            "d_query_torch": d_query_torch_np.flatten(),
            "query": query_np.flatten(),
        })
        df["output_rel"] = _rel_err_col(df, "output_diff", "output_torch")
        df["d_query_rel"] = _rel_err_col(df, "d_query_diff", "d_query_torch")
        abs_score = _max_abs_score(df, ["output_diff", "d_query_diff"])
        rel_score = _max_rel_score(
            df,
            [("output_diff", "output_torch"), ("d_query_diff", "d_query_torch")],
        )
        df_abs = df.copy()
        df_abs["abs_score"] = abs_score
        df_rel = df.copy()
        df_rel["rel_score"] = rel_score
        df_abs = _topk_rows_by_score(df_abs, df_abs["abs_score"])
        df_rel = _topk_rows_by_score(df_rel, df_rel["rel_score"])

        b_kv, s_kv, h_kv, d_kv = np.indices(key_np.shape)
        df_kv = pd.DataFrame({
            "batch": b_kv.flatten(),
            "seqlen": s_kv.flatten(),
            "head": h_kv.flatten(),
            "d": d_kv.flatten(),
            "d_key_diff": d_key_diff_np.flatten(),
            "d_key": d_key_np.flatten(),
            "d_key_torch": d_key_torch_np.flatten(),
            "d_value_diff": d_value_diff_np.flatten(),
            "d_value": d_value_np.flatten(),
            "d_value_torch": d_value_torch_np.flatten(),
            "key": key_np.flatten(),
            "value": value_np.flatten(),
        })
        df_kv["d_key_rel"] = _rel_err_col(df_kv, "d_key_diff", "d_key_torch")
        df_kv["d_value_rel"] = _rel_err_col(df_kv, "d_value_diff", "d_value_torch")
        kv_abs_score = _max_abs_score(df_kv, ["d_key_diff", "d_value_diff"])
        kv_rel_score = _max_rel_score(
            df_kv,
            [("d_key_diff", "d_key_torch"), ("d_value_diff", "d_value_torch")],
        )
        df_kv_abs = df_kv.copy()
        df_kv_abs["abs_score"] = kv_abs_score
        df_kv_rel = df_kv.copy()
        df_kv_rel["rel_score"] = kv_rel_score
        df_kv_abs = _topk_rows_by_score(df_kv_abs, df_kv_abs["abs_score"])
        df_kv_rel = _topk_rows_by_score(df_kv_rel, df_kv_rel["rel_score"])

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = "/outputs"
        os.makedirs(out_dir, exist_ok=True)
        excel_path = (
            f"{out_dir}/{now}_bwd_b_{batch_size}_hq_{nheads}_hk_{nheads_k}"
            f"_seqlen_q_{seqlen_q}_seqlen_k_{seqlen_k}_hdim_{d}_causal_{causal}.xlsx"
        )
        with pd.ExcelWriter(excel_path) as writer:
            df_abs.to_excel(writer, sheet_name="output_dq_abs", index=False)
            df_rel.to_excel(writer, sheet_name="output_dq_rel", index=False)
            df_kv_abs.to_excel(writer, sheet_name="dk_dv_kv_abs", index=False)
            df_kv_rel.to_excel(writer, sheet_name="dk_dv_kv_rel", index=False)
        print(f"Saved Excel debug file (failure case): {excel_path}")

    _assert_metrics(output_metrics, name="output", **bwd_tols)
    _assert_metrics(dq_metrics, name="dQ", **bwd_tols)
    _assert_metrics(dk_metrics, name="dK", **bwd_tols)
    _assert_metrics(dv_metrics, name="dV", **bwd_tols)


def _pack_padded_tensor(x, seqlens):
    chunks = []
    total = 0
    for i, seqlen in enumerate(seqlens):
        seqlen_val = int(seqlen.item())
        if seqlen_val <= 0:
            continue
        chunks.append(x[i, :seqlen_val].contiguous())
        total += seqlen_val
    if not chunks:
        return x.new_zeros((0,) + x.shape[2:])
    packed = torch.cat(chunks, dim=0)
    assert packed.shape[0] == total
    return packed




@pytest.mark.parametrize("dtype", [torch.float16])
# @pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("batch_size", [1, 3])
# @pytest.mark.parametrize("nheads, nheads_k", [(1, 1)])
@pytest.mark.parametrize("nheads, nheads_k", [(2, 1), (4, 2), (6, 3), (6, 1)])
# @pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "max_seqlen_q, max_seqlen_k", 
    [
        # (4, 4),
        # (64, 64),
        # (64, 128),
        # (64, 256),
        # (128, 64),
        # (256, 64),
        # (128, 128),
        # (1024, 1024),
        # (128, 256),
        # (128, 1024),
        # (256, 1024),
        # (512, 1024),
        # (256, 128),
        # (512, 128),
        # (768, 128),
        # (1024, 128),
        # (1024, 256),
        # (63, 63),
        # (65, 65),
        # (127, 127),
        # (129, 129),
        # (1, 1),
        # (1, 2),
        # (2, 1),
        # (2, 2),       
        # (64, 128),
        # (64, 256),
        # (128, 64),
        # (256, 64),
        # (128, 128),
        # (1024, 1024),
        # (128, 256),
        # (128, 1024),
        # (256, 1024),
        # (512, 1024),
        # (256, 128),
        # (512, 128),
        # (768, 128),
        # (1024, 128),
        # (1024, 256),
        # (64, 2),
        # (127, 63),
        # (129, 65),
        # (128, 127),
        # (128, 129),
        # (128, 1025),
        # (256, 1025),
        # (128, 128),
        # (1024, 1024),
        # (128, 256),
        # (256, 64),
        # (897, 1024),
        # (959, 1024),
        # (960, 1024),
        # (961, 1024),
        # (1023, 1024),
        # (1024, 1023),
        # (1024, 897),
        # (1,64),
        # (1,128),
        # (65,64),
        # (65,128),
        # (129,64),
        # (129,128),
        # (257,64),
        # (257,128),
        # (1, 1024),
        # (1023, 1024),
        # (1025, 1024),
        (64, 1),
        # (128,1),
        # (64, 65),
        # (128,65),
        # (64, 129),
        # (128,129),
        # (64, 257),
        # (128,257),
        # (1024, 1),
        # (1024, 1023),
        # (1024, 1025),
    ]
)
def test_flash_attn_bwd_varlen(
    batch_size,
    nheads,
    nheads_k,
    max_seqlen_q,
    max_seqlen_k,
    d,
    causal,
    dtype,
    use_identity_inputs=False,
):
    torch.cuda.init()
    torch.cuda.current_device()
    device = "cuda"

    seqlens_q = torch.randint(1, max_seqlen_q + 1, (batch_size,), dtype=torch.int32)
    seqlens_k = torch.randint(1, max_seqlen_k + 1, (batch_size,), dtype=torch.int32)

    if batch_size > 0:
        idx_q = torch.randint(0, batch_size, (1,), dtype=torch.int64).item()
        if batch_size > 1:
            idx_k = torch.randint(0, batch_size - 1, (1,), dtype=torch.int64).item()
            if idx_k >= idx_q:
                idx_k += 1
        else:
            idx_k = idx_q
        seqlens_q[idx_q] = max_seqlen_q
        seqlens_k[idx_k] = max_seqlen_k

    # seqlens_q = torch.tensor([3, 3, 4, 2], dtype=torch.int32)
    # seqlens_k = torch.tensor([3, 4, 2, 3], dtype=torch.int32)

    cu_seqlens_q = torch.cat(
        [torch.tensor([0], dtype=torch.int32), torch.cumsum(seqlens_q, dim=0, dtype=torch.int32)]
    ).to(device=device, dtype=torch.int32)
    cu_seqlens_k = torch.cat(
        [torch.tensor([0], dtype=torch.int32), torch.cumsum(seqlens_k, dim=0, dtype=torch.int32)]
    ).to(device=device, dtype=torch.int32)

    total_q = int(cu_seqlens_q[-1].item())
    total_k = int(cu_seqlens_k[-1].item())

    q_packed = torch.randn(total_q, nheads, d, device=device, dtype=dtype)
    k_packed = torch.randn(total_k, nheads_k, d, device=device, dtype=dtype)
    v_packed = torch.randn(total_k, nheads_k, d, device=device, dtype=dtype)

    q_padded = torch.zeros(batch_size, max_seqlen_q, nheads, d, device=device, dtype=dtype)
    k_padded = torch.zeros(batch_size, max_seqlen_k, nheads_k, d, device=device, dtype=dtype)
    v_padded = torch.zeros(batch_size, max_seqlen_k, nheads_k, d, device=device, dtype=dtype)

    if use_identity_inputs:
        q_padded.zero_()
        k_padded.zero_()
        v_padded.zero_()

    offset_q = 0
    for i, seqlen in enumerate(seqlens_q.tolist()):
        next_offset = offset_q + seqlen
        q_padded[i, :seqlen] = q_packed[offset_q:next_offset]
        offset_q = next_offset

    offset_k = 0
    for i, seqlen in enumerate(seqlens_k.tolist()):
        next_offset = offset_k + seqlen
        k_padded[i, :seqlen] = k_packed[offset_k:next_offset]
        v_padded[i, :seqlen] = v_packed[offset_k:next_offset]
        offset_k = next_offset

    if use_identity_inputs:
        _fill_identity_padded_(q_padded, seqlens_q)
        _fill_identity_padded_(k_padded, seqlens_k)
        _fill_identity_padded_(v_padded, seqlens_k)
        seqlens_q_cpu = seqlens_q.cpu()
        seqlens_k_cpu = seqlens_k.cpu()
        q_packed = _make_identity_packed(seqlens_q_cpu, nheads, d, device, dtype)
        k_packed = _make_identity_packed(seqlens_k_cpu, nheads_k, d, device, dtype)
        v_packed = _make_identity_packed(seqlens_k_cpu, nheads_k, d, device, dtype)

    if use_identity_inputs:
        d_output_padded = torch.zeros(batch_size, max_seqlen_q, nheads, d, device=device, dtype=dtype)
        _fill_identity_padded_(d_output_padded, seqlens_q)
        d_output_packed = _make_identity_packed(seqlens_q_cpu, nheads, d, device, dtype)
    else:
        d_output_padded = torch.randn(batch_size, max_seqlen_q, nheads, d, device=device, dtype=dtype)
        d_output_packed = _pack_padded_tensor(d_output_padded, seqlens_q)

    q_packed_flash = q_packed.clone().detach().requires_grad_(True)
    k_packed_flash = k_packed.clone().detach().requires_grad_(True)
    v_packed_flash = v_packed.clone().detach().requires_grad_(True)

    out_packed_flash = flash_attn_varlen_func(
        q_packed_flash,
        k_packed_flash,
        v_packed_flash,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal=causal,
    )
    torch.cuda.synchronize()

    grad_output_packed = d_output_packed.contiguous()
    dq_packed, dk_packed, dv_packed = torch.autograd.grad(
        outputs=out_packed_flash,
        inputs=(q_packed_flash, k_packed_flash, v_packed_flash),
        grad_outputs=grad_output_packed,
        retain_graph=False,
        allow_unused=False,
    )
    torch.cuda.synchronize()

    out_packed = out_packed_flash.detach()
    dq_packed = dq_packed.detach()
    dk_packed = dk_packed.detach()
    dv_packed = dv_packed.detach()

    output_ref = []
    dq_ref = []
    dk_ref = []
    dv_ref = []

    for i, (seqlen_q_i, seqlen_k_i) in enumerate(zip(seqlens_q.tolist(), seqlens_k.tolist())):
        q_i = q_padded[i:i + 1, :seqlen_q_i].contiguous()
        k_i = k_padded[i:i + 1, :seqlen_k_i].contiguous()
        v_i = v_padded[i:i + 1, :seqlen_k_i].contiguous()
        d_out_i = d_output_padded[i:i + 1, :seqlen_q_i].contiguous()

        out_i, dq_i, dk_i, dv_i = vanilla_attention_ref(
            q_i,
            k_i,
            v_i,
            d_out_i,
            causal,
        )
        output_ref.append(out_i.squeeze(0))
        dq_ref.append(dq_i.squeeze(0))
        dk_ref.append(dk_i.squeeze(0))
        dv_ref.append(dv_i.squeeze(0))

    output_ref_packed = torch.cat(output_ref, dim=0)
    dq_ref_packed = torch.cat(dq_ref, dim=0)
    dk_ref_packed = torch.cat(dk_ref, dim=0)
    dv_ref_packed = torch.cat(dv_ref, dim=0)

    out_packed = out_packed.detach().clone()
    dq_packed = dq_packed.detach().clone()
    dk_packed = dk_packed.detach().clone()
    dv_packed = dv_packed.detach().clone()

    # print("==========")
    # print("==========")
    # print("==========")
    # print("==========")
    # print("==========")
    # print(f"seqlen_q = {seqlens_q}, seqlens_k = {seqlens_k}")
    # print(f"q_packed = {q_packed}")
    # print(f"seqlen_q = {seqlens_q}, seqlens_k = {seqlens_k}")
    # print(f"k_packed = {k_packed}")
    # print(f"seqlens_q = {seqlens_q}, seqlens_k = {seqlens_k}")
    # print(f"v_packed = {v_packed}")
    # print(f"seqlens_q = {seqlens_q}, seqlens_k = {seqlens_k}")
    # print(f"seqlens_q = {seqlens_q}, seqlens_k = {seqlens_k}")
    # print(f"out = {out_packed}, output_ref = {output_ref_packed}")
    # print(f"seqlens_q = {seqlens_q}, seqlens_k = {seqlens_k}")
    # print(f"dq = {dq_packed}, dq_ref = {dq_ref_packed}")
    # print(f"seqlens_q = {seqlens_q}, seqlens_k = {seqlens_k}")
    # print(f"dk = {dk_packed}, dk_ref = {dk_ref_packed}")
    # print(f"seqlens_q = {seqlens_q}, seqlens_k = {seqlens_k}")
    # print(f"dv = {dv_packed}, dv_ref = {dv_ref_packed}")
  
    output_metrics = _error_metrics(out_packed, output_ref_packed)
    dq_metrics = _error_metrics(dq_packed, dq_ref_packed)
    dk_metrics = _error_metrics(dk_packed, dk_ref_packed)
    dv_metrics = _error_metrics(dv_packed, dv_ref_packed)

    print("\n")
    print("========================================")
    print(
        "params: "
        f"batch_size={batch_size}, "
        f"nheads={nheads}, "
        f"nheads_k={nheads_k}, "
        f"max_seqlen_q={max_seqlen_q}, "
        f"max_seqlen_k={max_seqlen_k}, "
        f"d={d}, "
        f"causal={causal}, "
        f"dtype={dtype}, "
        f"seqlens_q={seqlens_q.tolist()}, "
        f"seqlens_k={seqlens_k.tolist()}"
    )
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
    print("========================================\n\n\n")


    bwd_tols = BWD_TOLS

    failed = any(
        not (
            m["max_abs"] <= bwd_tols["atol"] and
            m["max_rel"] <= bwd_tols["rtol"] and
            m["l2_rel"] <= bwd_tols["rtol_l2"] and
            m["mean_abs"] <= bwd_tols["mean_atol"] and
            m["mean_rel"] <= bwd_tols["mean_rtol"] and
            m["rms_rel"] <= bwd_tols["mean_rtol_l2"]
        )
        for m in (output_metrics, dq_metrics, dk_metrics, dv_metrics)
    )

    if failed and SAVE_FAIL_DEBUG_EXCEL:
        seqlens_q_list = [int(x) for x in seqlens_q.tolist()]
        seqlens_k_list = [int(x) for x in seqlens_k.tolist()]

        def _token_meta(seqlens):
            if not seqlens:
                return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
            batches = [np.full(seqlen, b, dtype=np.int32) for b, seqlen in enumerate(seqlens)]
            positions = [np.arange(seqlen, dtype=np.int32) for seqlen in seqlens]
            return (
                np.concatenate(batches, axis=0) if batches else np.array([], dtype=np.int32),
                np.concatenate(positions, axis=0) if positions else np.array([], dtype=np.int32),
            )

        token_batch_q, token_pos_q = _token_meta(seqlens_q_list)
        token_batch_k, token_pos_k = _token_meta(seqlens_k_list)

        output_diff_np = (out_packed - output_ref_packed).detach().cpu().numpy()
        output_np = out_packed.detach().cpu().numpy()
        output_ref_np = output_ref_packed.detach().cpu().numpy()

        dq_diff_np = (dq_packed - dq_ref_packed).detach().cpu().numpy()
        dq_np = dq_packed.detach().cpu().numpy()
        dq_ref_np = dq_ref_packed.detach().cpu().numpy()
        d_output_np = d_output_packed.detach().cpu().numpy()

        dk_diff_np = (dk_packed - dk_ref_packed).detach().cpu().numpy()
        dk_np = dk_packed.detach().cpu().numpy()
        dk_ref_np = dk_ref_packed.detach().cpu().numpy()

        dv_diff_np = (dv_packed - dv_ref_packed).detach().cpu().numpy()
        dv_np = dv_packed.detach().cpu().numpy()
        dv_ref_np = dv_ref_packed.detach().cpu().numpy()

        query_np = q_packed.detach().cpu().numpy()
        key_np = k_packed.detach().cpu().numpy()
        value_np = v_packed.detach().cpu().numpy()

        total_q, nheads, d = output_np.shape
        total_k = key_np.shape[0]
        nheads_k = key_np.shape[1]

        head_idx = np.tile(np.repeat(np.arange(nheads, dtype=np.int32), d), total_q)
        d_idx = np.tile(np.arange(d, dtype=np.int32), total_q * nheads)
        batch_idx = np.repeat(token_batch_q, nheads * d)
        seqlen_idx = np.repeat(token_pos_q, nheads * d)

        df = pd.DataFrame({
            "batch": batch_idx,
            "seqlen": seqlen_idx,
            "head": head_idx,
            "d": d_idx,
            "output_diff": output_diff_np.reshape(-1),
            "output": output_np.reshape(-1),
            "output_ref": output_ref_np.reshape(-1),
            "d_query_diff": dq_diff_np.reshape(-1),
            "d_query": dq_np.reshape(-1),
            "d_query_ref": dq_ref_np.reshape(-1),
            "query": query_np.reshape(-1),
            "d_output": d_output_np.reshape(-1),
        })
        df["output_rel"] = _rel_err_col(df, "output_diff", "output_ref")
        df["d_query_rel"] = _rel_err_col(df, "d_query_diff", "d_query_ref")
        abs_score = _max_abs_score(df, ["output_diff", "d_query_diff"])
        rel_score = _max_rel_score(
            df,
            [("output_diff", "output_ref"), ("d_query_diff", "d_query_ref")],
        )
        df_abs = df.copy()
        df_abs["abs_score"] = abs_score
        df_rel = df.copy()
        df_rel["rel_score"] = rel_score
        df_abs = _topk_rows_by_score(df_abs, df_abs["abs_score"])
        df_rel = _topk_rows_by_score(df_rel, df_rel["rel_score"])

        head_k_idx = np.tile(np.repeat(np.arange(nheads_k, dtype=np.int32), d), total_k)
        d_k_idx = np.tile(np.arange(d, dtype=np.int32), total_k * nheads_k)
        batch_k_idx = np.repeat(token_batch_k, nheads_k * d)
        seqlen_k_idx = np.repeat(token_pos_k, nheads_k * d)

        df_kv = pd.DataFrame({
            "batch": batch_k_idx,
            "seqlen": seqlen_k_idx,
            "head": head_k_idx,
            "d": d_k_idx,
            "d_key_diff": dk_diff_np.reshape(-1),
            "d_key": dk_np.reshape(-1),
            "d_key_ref": dk_ref_np.reshape(-1),
            "d_value_diff": dv_diff_np.reshape(-1),
            "d_value": dv_np.reshape(-1),
            "d_value_ref": dv_ref_np.reshape(-1),
            "key": key_np.reshape(-1),
            "value": value_np.reshape(-1),
        })
        df_kv["d_key_rel"] = _rel_err_col(df_kv, "d_key_diff", "d_key_ref")
        df_kv["d_value_rel"] = _rel_err_col(df_kv, "d_value_diff", "d_value_ref")
        kv_abs_score = _max_abs_score(df_kv, ["d_key_diff", "d_value_diff"])
        kv_rel_score = _max_rel_score(
            df_kv,
            [("d_key_diff", "d_key_ref"), ("d_value_diff", "d_value_ref")],
        )
        df_kv_abs = df_kv.copy()
        df_kv_abs["abs_score"] = kv_abs_score
        df_kv_rel = df_kv.copy()
        df_kv_rel["rel_score"] = kv_rel_score
        df_kv_abs = _topk_rows_by_score(df_kv_abs, df_kv_abs["abs_score"])
        df_kv_rel = _topk_rows_by_score(df_kv_rel, df_kv_rel["rel_score"])

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = "/outputs"
        os.makedirs(out_dir, exist_ok=True)
        excel_path = (
            f"{out_dir}/{now}_bwd_varlen_b_{batch_size}_hq_{nheads}_hk_{nheads_k}"
            f"_maxq_{max_seqlen_q}_maxk_{max_seqlen_k}_hdim_{d}_causal_{causal}.xlsx"
        )
        with pd.ExcelWriter(excel_path) as writer:
            df_abs.to_excel(writer, sheet_name="output_dq_abs", index=False)
            df_rel.to_excel(writer, sheet_name="output_dq_rel", index=False)
            df_kv_abs.to_excel(writer, sheet_name="dk_dv_kv_abs", index=False)
            df_kv_rel.to_excel(writer, sheet_name="dk_dv_kv_rel", index=False)
        print(f"Saved Excel debug file (failure case): {excel_path}")

    _assert_metrics(output_metrics, name="output", **bwd_tols)
    _assert_metrics(dq_metrics, name="dQ", **bwd_tols)
    _assert_metrics(dk_metrics, name="dK", **bwd_tols)
    _assert_metrics(dv_metrics, name="dV", **bwd_tols)
