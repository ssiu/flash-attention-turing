
import pytest
import torch
import os
from datetime import datetime

import torch.nn.functional as F
from torch.nn.attention import bias
import pandas as pd
import numpy as np

from flash_attn_turing import (
    fwd,
    bwd
)

torch.set_printoptions(threshold=torch.inf)

EXCEL_TOPK_ROWS = 10_000
EXCEL_REL_EPS = 1e-6
TEST_REL_EPS = 1e-6
SAVE_FAIL_DEBUG_EXCEL = True


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


def _assert_metrics(metrics, atol, rtol, rtol_l2, mean_atol, mean_rtol, mean_rtol_l2, name):
    assert metrics["max_abs"] <= atol, f"{name} max_abs={metrics['max_abs']} > atol={atol}"
    assert metrics["max_rel"] <= rtol, f"{name} max_rel={metrics['max_rel']} > rtol={rtol}"
    assert metrics["l2_rel"] <= rtol_l2, f"{name} l2_rel={metrics['l2_rel']} > rtol_l2={rtol_l2}"
    assert metrics["mean_abs"] <= mean_atol, f"{name} mean_abs={metrics['mean_abs']} > mean_atol={mean_atol}"
    assert metrics["mean_rel"] <= mean_rtol, f"{name} mean_rel={metrics['mean_rel']} > mean_rtol={mean_rtol}"
    assert metrics["rms_rel"] <= mean_rtol_l2, f"{name} rms_rel={metrics['rms_rel']} > mean_rtol_l2={mean_rtol_l2}"

def create_stacked_identity_matrix(batch_size, seqlen, nheads, d, device, dtype):
    """
    Creates a tensor with stacked identity matrices along the seqlen_q dimension.

    Args:
        batch_size (int): The batch size.
        seqlen_q (int): The sequence length, which must be divisible by d.
        nheads (int): The number of heads.
        d (int): The dimension of the identity matrix (64 or 128).
        device (torch.device): The device to create the tensor on.
        dtype (torch.dtype): The data type of the tensor.

    Returns:
        torch.Tensor: The resulting tensor of shape (batch_size, seqlen_q, nheads, d).
    """
    if seqlen % d != 0:
        raise ValueError("seqlen_q must be divisible by d")

    # Create a single identity matrix of size d x d
    identity_matrix = torch.eye(d, device=device, dtype=dtype)

    # Calculate how many times to stack the identity matrix
    num_stacks = seqlen // d

    # Stack the identity matrix along the first dimension
    # This creates a tensor of shape (seqlen_q, d)
    stacked_matrix = identity_matrix.repeat(num_stacks, 1)

    # Reshape and expand dimensions to match the target tensor shape
    # (batch_size, seqlen_q, nheads, d)
    stacked_tensor_view = stacked_matrix.view(1, seqlen, 1, d)
    final_tensor = stacked_tensor_view.expand(batch_size, seqlen, nheads, d)

    return final_tensor

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


# def attention_ref(query, key, value, d_output, causal=False):
#     query_torch = query.permute(0, 2, 1, 3).contiguous().clone()
#     key_torch = key.permute(0, 2, 1, 3).contiguous().clone()
#     value_torch = value.permute(0, 2, 1, 3).contiguous().clone()

#     query_torch.requires_grad = True
#     key_torch.requires_grad = True
#     value_torch.requires_grad = True


#     batch_size, nheads, seqlen_q,  d = query_torch.size()
#     _ , _ , seqlen_k,  _ = key_torch.size()

#     scores = torch.matmul(query_torch, key_torch.transpose(-2, -1)) / (d ** 0.5)
    
#     if causal:
#         #attn_mask = torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=attn.device).tril(diagonal=0)

#         attn_mask = causal_lower_right(seqlen_q, seqlen_k, device=scores.device)
#         attn_mask = attn_mask.view(1, 1, seqlen_q, seqlen_k)     # broadcast over batch & heads
#         scores = scores.masked_fill(~attn_mask, float("-inf"))
#         # mask_val = torch.finfo(attn.dtype).min / 2
#         # attn = attn.masked_fill(~attn_mask, mask_val)
#         #attn = attn.masked_fill(~attn_mask, -1e9)




#     attn = F.softmax(scores, dim=-1)
#     attn = torch.where(attn.isnan(), torch.zeros_like(attn), attn)

#     # if causal:
#     #     for i in range(32):
#     #         print(f"attention matrix {attn[0, 0, 0, i]}")
#     output_torch = torch.matmul(attn, value_torch)


#     d_value_torch = torch.matmul(attn.transpose(2,3), d_output_torch)

#     output_torch = output_torch.permute(0, 2, 1, 3).contiguous().clone()




#     d_query_torch = query_torch.grad.permute(0, 2, 1, 3).contiguous().clone()
#     d_key_torch = key_torch.grad.permute(0, 2, 1, 3).contiguous().clone()

#     d_value = d_value_torch.permute(0, 2, 1, 3).contiguous().clone()

#     return output_torch

# MY OWN attn ref code

# def attention_ref(query, key, value, d_output=None, causal=False):
#     query_torch = query.permute(0, 2, 1, 3).contiguous().clone()
#     key_torch = key.permute(0, 2, 1, 3).contiguous().clone()
#     value_torch = value.permute(0, 2, 1, 3).contiguous().clone()

#     query_torch.requires_grad = True
#     key_torch.requires_grad = True
#     value_torch.requires_grad = True


#     query_torch = query_torch.detach().clone().requires_grad_(True)
#     key_torch   = key_torch.detach().clone().requires_grad_(True)
#     value_torch = value_torch.detach().clone().requires_grad_(True)

#     assert query_torch.requires_grad and query_torch.is_leaf
#     assert key_torch.requires_grad and key_torch.is_leaf
#     assert value_torch.requires_grad and value_torch.is_leaf

#     batch_size, nheads, seqlen_q,  d = query_torch.size()
#     _ , _ , seqlen_k,  _ = key_torch.size()

#     scores = torch.matmul(query_torch, key_torch.transpose(-2, -1)) / (d ** 0.5)
#     # print(f"attention_ref, query = {query}")
#     # print(f"attention_ref, key = {key}")
#     # print(f"attention_ref, scores = {scores}")

#     if causal:
#         #attn_mask = torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=attn.device).tril(diagonal=0)

#         attn_mask = causal_lower_right(seqlen_q, seqlen_k, device=scores.device)
#         attn_mask = attn_mask.view(1, 1, seqlen_q, seqlen_k)     # broadcast over batch & heads
#         scores = scores.masked_fill(~attn_mask, float("-inf"))
#         # mask_val = torch.finfo(attn.dtype).min / 2
#         # attn = attn.masked_fill(~attn_mask, mask_val)
#         #attn = attn.masked_fill(~attn_mask, -1e9)




#     attn = F.softmax(scores, dim=-1)
#     attn = torch.where(attn.isnan(), torch.zeros_like(attn), attn)

#     # print(f"attention_ref, attn = {attn}")
#     # print(f"attention_ref, value = {value_torch}")
#     # if causal:
#     #     for i in range(32):
#     #         print(f"attention matrix {attn[0, 0, 0, i]}")
#     output_torch = torch.matmul(attn, value_torch)

#     if d_output is None:
#         output_torch = output_torch.permute(0, 2, 1, 3).contiguous().clone()
#         return output_torch
#     else:
        
#         d_output_torch = d_output.permute(0, 2, 1, 3).contiguous()
#         grads = torch.autograd.grad(
#             outputs=output_torch,
#             inputs=(query_torch, key_torch, value_torch),
#             grad_outputs=d_output_torch,
#             retain_graph=False,
#             allow_unused=True
#         )

#         d_query_torch, d_key_torch, d_value_torch = grads

        
#         output_torch = output_torch.permute(0, 2, 1, 3).contiguous().clone()
#         d_query_torch = d_query_torch.permute(0, 2, 1, 3).contiguous().clone()
#         d_key_torch = d_key_torch.permute(0, 2, 1, 3).contiguous().clone()
#         d_value_torch = d_value_torch.permute(0, 2, 1, 3).contiguous().clone()



#         return output_torch, d_query_torch, d_key_torch, d_value_torch



def attention_ref(query, key, value, d_output=None, causal=False):
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
# @pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("nheads, nheads_k", [(1, 1), (8, 8), (8, 2), (8, 1)])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 1),
        (1, 2),
        # (1, 64),
        # (1, 128),
        # (2, 1),
        # (2, 64),
        # (2, 128),
        # (2,2),
        # (64,1),
        # (127,1),
        # (128,1),
        # (129,1),
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
        # (1024, 897)
    ],
)
# @pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("causal", [False])
def test_flash_attn_fwd(
        batch_size, nheads, nheads_k, seqlen_q, seqlen_k,  d, causal, dtype
):
    device = "cuda"


    # batch_size = 2
    # nheads = 1
    query = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    key = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)
    value = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype)

    output, l = fwd(query, key, value, causal)



    output_torch = attention_ref(query, key, value, None, causal)

    output_metrics = _error_metrics(output, output_torch)
    idx = torch.argmax((output - output_torch).abs())
    coords = np.unravel_index(idx.item(), (output - output_torch).shape)

    output_max = output.abs().max().item()
    output_mean = output.abs().mean().item()

    output_torch_max = output_torch.abs().max().item()
    output_torch_mean = output_torch.abs().mean().item()

    print(f"seqlen_q = {seqlen_q}, seqlen_k = {seqlen_k}, d = {d}, causal = {causal}")

    print(f"output max: {output_max}")
    print(f"output mean: {output_mean}")
    print(f"coordinate: {coords}")

    print(f"output_torch max: {output_torch_max}")
    print(f"output_torch mean: {output_torch_mean}")

    print(f"output max abs diff: {output_metrics['max_abs']}")
    print(f"output mean abs diff: {output_metrics['mean_abs']}")
    print(f"output max rel diff: {output_metrics['max_rel']}")
    print(f"output mean rel diff: {output_metrics['mean_rel']}")
    print(f"output l2 rel diff: {output_metrics['l2_rel']}")
    print(f"output rms rel diff: {output_metrics['rms_rel']}")

    save_excel = os.getenv("SAVE_DEBUG_EXCEL", "0") == "1"
    if save_excel:
        output_diff_np = (output - output_torch).detach().cpu().numpy()
        output_np = output.detach().cpu().numpy()
        output_torch_np = output_torch.detach().cpu().numpy()
        query_np = query.detach().cpu().numpy()

        b_idx, s_idx, h_idx, d_idx = np.indices(output_diff_np.shape)
        df = pd.DataFrame({
            "batch": b_idx.flatten(),
            "seqlen": s_idx.flatten(),
            "head": h_idx.flatten(),
            "d": d_idx.flatten(),
            "diff": output_diff_np.flatten(),
            "output": output_np.flatten(),
            "output_torch": output_torch_np.flatten(),
            "query": query_np.flatten(),
        })
        df["diff_rel"] = _rel_err_col(df, "diff", "output_torch")
        abs_score = _max_abs_score(df, ["diff"])
        rel_score = _max_rel_score(df, [("diff", "output_torch")])
        df_abs = df.copy()
        df_abs["abs_score"] = abs_score
        df_rel = df.copy()
        df_rel["rel_score"] = rel_score
        df_abs = _topk_rows_by_score(df_abs, df_abs["abs_score"])
        df_rel = _topk_rows_by_score(df_rel, df_rel["rel_score"])

        l_np = l.detach().cpu().numpy()
        if l_np.shape == (batch_size, nheads, seqlen_q):
            b_l, h_l, s_l = np.indices(l_np.shape)
            l_df = pd.DataFrame({
                "batch": b_l.flatten(),
                "head": h_l.flatten(),
                "seqlen": s_l.flatten(),
                "l": l_np.flatten(),
            })
        elif l_np.shape == (batch_size, seqlen_q, nheads):
            b_l, s_l, h_l = np.indices(l_np.shape)
            l_df = pd.DataFrame({
                "batch": b_l.flatten(),
                "seqlen": s_l.flatten(),
                "head": h_l.flatten(),
                "l": l_np.flatten(),
            })
        elif l_np.shape == (batch_size, seqlen_q):
            b_l, s_l = np.indices(l_np.shape)
            l_df = pd.DataFrame({
                "batch": b_l.flatten(),
                "seqlen": s_l.flatten(),
                "l": l_np.flatten(),
            })
        else:
            dim_idx = np.indices(l_np.shape)
            l_cols = {f"dim{i}": dim_idx[i].flatten() for i in range(l_np.ndim)}
            l_df = pd.DataFrame(l_cols)
            l_df["l"] = l_np.flatten()

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = "outputs"
        os.makedirs(out_dir, exist_ok=True)
        excel_path = (
            f"{out_dir}/{now}_fwd_b_{batch_size}_hq_{nheads}_hk_{nheads_k}"
            f"_seqlen_q_{seqlen_q}_seqlen_k_{seqlen_k}_hdim_{d}_causal_{causal}.xlsx"
        )
        with pd.ExcelWriter(excel_path) as writer:
            df_abs.to_excel(writer, sheet_name="output_diff_abs", index=False)
            df_rel.to_excel(writer, sheet_name="output_diff_rel", index=False)
            l_df.to_excel(writer, sheet_name="l_values", index=False)


    _assert_metrics(
        output_metrics,
        atol=1e-2,
        rtol=1e-1,
        rtol_l2=2e-2,
        mean_atol=1e-3,
        mean_rtol=1e-2,
        mean_rtol_l2=1e-2,
        name="output",
    )

def generate_tensor_bwd_64_64():
    torch.cuda.init()
    batch_size = 1
    seqlen_q = 64
    seqlen_k = 64
    nheads = 1
    d = 64
    device = "cuda"
    dtype = torch.float16

    query = torch.zeros(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    key = torch.zeros(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    value = torch.zeros(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    d_output = torch.zeros(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)

    eye64 = torch.eye(64, device=device, dtype=dtype)

    # Single 64x64 block
    query[0, :, 0, :] = eye64
    d_output[0, :, 0, :] = eye64
    key[0, :, 0, :] = eye64
    value[0, :, 0, :] = eye64

    return query, key, value, d_output

def generate_tensor_bwd_128_128():
    torch.cuda.init()
    batch_size = 1
    seqlen_q = 128
    seqlen_k = 128
    nheads = 1
    d = 128
    device = "cuda"
    dtype = torch.float16

    query = torch.zeros(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    key = torch.zeros(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    value = torch.zeros(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    d_output = torch.zeros(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)

    eye128 = torch.eye(128, device=device, dtype=dtype)

    # Single 64x64 block
    query[0, :, 0, :] = eye128
    d_output[0, :, 0, :] = eye128
    key[0, :, 0, :] = eye128
    value[0, :, 0, :] = eye128

    return query, key, value, d_output



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

    torch.set_printoptions(
        threshold=1_000_000,  # big enough to avoid truncation
        linewidth=300,
        precision=1,
        sci_mode=False
    )


    # output_torch,  d_query_torch, d_key_torch, d_value_torch = attention_ref(query, key, value, d_output, causal)

    output_torch,  d_query_torch, d_key_torch, d_value_torch = memory_efficient_attention_ref(query, key, value, d_output, causal)


    torch.cuda.synchronize()


    output_torch = output_torch.detach().clone()
    d_query_torch = d_query_torch.detach().clone()
    d_key_torch = d_key_torch.detach().clone()
    d_value_torch = d_value_torch.detach().clone()

    output, l = fwd(query, key, value, causal)
    torch.cuda.synchronize()
    d_query, d_key, d_value = bwd(query, key, value, output, l, d_output, causal)
    torch.cuda.synchronize()

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


    bwd_tols = dict(
        atol=5e-3,
        rtol=1000,
        rtol_l2=100,
        mean_atol=2e-4,
        mean_rtol=1e-2,
        mean_rtol_l2=100,
    )
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

        l_np = l.detach().cpu().numpy()
        if l_np.shape == (batch_size, nheads, seqlen_q):
            b_l, h_l, s_l = np.indices(l_np.shape)
            l_df = pd.DataFrame({
                "batch": b_l.flatten(),
                "head": h_l.flatten(),
                "seqlen": s_l.flatten(),
                "l": l_np.flatten(),
            })
        elif l_np.shape == (batch_size, seqlen_q, nheads):
            b_l, s_l, h_l = np.indices(l_np.shape)
            l_df = pd.DataFrame({
                "batch": b_l.flatten(),
                "seqlen": s_l.flatten(),
                "head": h_l.flatten(),
                "l": l_np.flatten(),
            })
        elif l_np.shape == (batch_size, seqlen_q):
            b_l, s_l = np.indices(l_np.shape)
            l_df = pd.DataFrame({
                "batch": b_l.flatten(),
                "seqlen": s_l.flatten(),
                "l": l_np.flatten(),
            })
        else:
            dim_idx = np.indices(l_np.shape)
            l_cols = {f"dim{i}": dim_idx[i].flatten() for i in range(l_np.ndim)}
            l_df = pd.DataFrame(l_cols)
            l_df["l"] = l_np.flatten()

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
            l_df.to_excel(writer, sheet_name="l_values", index=False)
        print(f"Saved Excel debug file (failure case): {excel_path}")

    _assert_metrics(output_metrics, name="output", **bwd_tols)
    _assert_metrics(dq_metrics, name="dQ", **bwd_tols)
    _assert_metrics(dk_metrics, name="dK", **bwd_tols)
    _assert_metrics(dv_metrics, name="dV", **bwd_tols)
