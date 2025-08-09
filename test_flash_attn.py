
import pytest
import torch

import torch.nn.functional as F

from flash_attn_turing import (
    fwd,
    bwd
)

def attention_ref(query, key, value, d_output, causal=False):
    query_torch = query.permute(0, 2, 1, 3).contiguous().clone()
    key_torch = key.permute(0, 2, 1, 3).contiguous().clone()
    value_torch = value.permute(0, 2, 1, 3).contiguous().clone()

    d_output_torch = d_output.permute(0, 2, 1, 3).contiguous().clone()

    query_torch.requires_grad = True
    key_torch.requires_grad = True
    value_torch.requires_grad = True

    output_torch = F.scaled_dot_product_attention(query=query_torch, key=key_torch, value=value_torch, is_causal=causal)


    output_torch.backward(d_output_torch)
    output_torch = output_torch.permute(0, 2, 1, 3).contiguous().clone()

    d_query_torch = query_torch.grad.permute(0, 2, 1, 3).contiguous().clone()
    d_key_torch = key_torch.grad.permute(0, 2, 1, 3).contiguous().clone()
    d_value_torch = value_torch.grad.permute(0, 2, 1, 3).contiguous().clone()

    return output_torch,  d_query_torch, d_key_torch, d_value_torch


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("seqlen",
                        [4096]
)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn(
        seqlen, d, causal, dtype
):
    device = "cuda"

    batch_size = 4
    nheads = 4
    query = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    key = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    value = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    d_output = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)

    output, l = fwd(query, key, value, causal)
    d_query, d_key, d_value = bwd(query, key, value, output, l, d_output, bool(is_causal))

    output_torch,  d_query_torch, d_key_torch, d_value_torch = attention_ref(query, key, value, d_output, causal)

    dq_max_diff = (d_query - d_query_torch).abs().max().item()
    dk_max_diff = (d_key - d_key_torch).abs().max().item()
    dv_max_diff = (d_value - d_value_torch).abs().max().item()

    dq_mean_diff = (d_query - d_query_torch).abs().mean().item()
    dk_mean_diff = (d_key - d_key_torch).abs().mean().item()
    dv_mean_diff = (d_value - d_value_torch).abs().mean().item()

    print(f"dQ max diff: {dq_max_diff}")
    print(f"dK max diff: {dk_max_diff}")
    print(f"dV max diff: {dv_max_diff}")
    print(f"dQ mean diff: {dq_mean_diff}")
    print(f"dK mean diff: {dk_mean_diff}")
    print(f"dV mean diff: {dv_mean_diff}")

    assert dq_max_diff <= 1e-3
    assert dk_max_diff <= 1e-3
    assert dv_max_diff <= 1e-3
    assert dq_mean_diff <= 1e-5
    assert dk_mean_diff <= 1e-5
    assert dv_mean_diff <= 1e-5


