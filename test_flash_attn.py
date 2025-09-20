
import pytest
import torch

import torch.nn.functional as F
from torch.nn.attention import bias

import numpy as np

from flash_attn_turing import (
    fwd,
    bwd
)


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


def attention_ref(query, key, value, d_output=None, causal=False):
    query_torch = query.permute(0, 2, 1, 3).contiguous().clone()
    key_torch = key.permute(0, 2, 1, 3).contiguous().clone()
    value_torch = value.permute(0, 2, 1, 3).contiguous().clone()
    



    query_torch.requires_grad = True
    key_torch.requires_grad = True
    value_torch.requires_grad = True



    batch_size, nheads, seqlen_q,  d = query_torch.size()
    _ , _ , seqlen_k,  _ = key_torch.size()

    scores = torch.matmul(query_torch, key_torch.transpose(-2, -1)) / (d ** 0.5)
    
    if causal:
        #attn_mask = torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=attn.device).tril(diagonal=0)

        attn_mask = causal_lower_right(seqlen_q, seqlen_k, device=scores.device)
        attn_mask = attn_mask.view(1, 1, seqlen_q, seqlen_k)     # broadcast over batch & heads
        scores = scores.masked_fill(~attn_mask, float("-inf"))
        # mask_val = torch.finfo(attn.dtype).min / 2
        # attn = attn.masked_fill(~attn_mask, mask_val)
        #attn = attn.masked_fill(~attn_mask, -1e9)




    attn = F.softmax(scores, dim=-1)
    attn = torch.where(attn.isnan(), torch.zeros_like(attn), attn)

    # if causal:
    #     for i in range(32):
    #         print(f"attention matrix {attn[0, 0, 0, i]}")
    output_torch = torch.matmul(attn, value_torch)

    if d_output is None:
        output_torch = output_torch.permute(0, 2, 1, 3).contiguous().clone()
        return output_torch
    else:
        
        d_output_torch = d_output.permute(0, 2, 1, 3).contiguous()
        grads = torch.autograd.grad(
            outputs=output_torch,
            inputs=(query_torch, key_torch, value_torch),
            grad_outputs=d_output_torch,
            retain_graph=False,
            allow_unused=True
        )

        d_query_torch, d_key_torch, d_value_torch = grads
        output_torch = output_torch.permute(0, 2, 1, 3).contiguous().clone()
        d_query_torch = d_query_torch.permute(0, 2, 1, 3).contiguous().clone()
        d_key_torch = d_key_torch.permute(0, 2, 1, 3).contiguous().clone()
        d_value_torch = d_value_torch.permute(0, 2, 1, 3).contiguous().clone()

        return output_torch, d_query_torch, d_key_torch, d_value_torch




def memory_efficient_attention_ref(query, key, value, d_output=None, causal=False):
    query_torch = query.permute(0, 2, 1, 3).contiguous().clone()
    key_torch = key.permute(0, 2, 1, 3).contiguous().clone()
    value_torch = value.permute(0, 2, 1, 3).contiguous().clone()

    d_output_torch = d_output.permute(0, 2, 1, 3).contiguous().clone()


    seqlen_q = query_torch.size(2)
    seqlen_k = key_torch.size(2)

    query_torch.requires_grad = True
    key_torch.requires_grad = True
    value_torch.requires_grad = True

    if causal:
        attn_mask = bias.causal_lower_right(seqlen_q, seqlen_k)
    else:
        attn_mask=None

    output_torch = F.scaled_dot_product_attention(query=query_torch, key=key_torch, value=value_torch, attn_mask=attn_mask)


    if d_output == None:
        return output_torch

    else:
        output_torch.backward(d_output_torch)
        output_torch = output_torch.permute(0, 2, 1, 3).contiguous().clone()

        d_query_torch = query_torch.grad.permute(0, 2, 1, 3).contiguous().clone()
        d_key_torch = key_torch.grad.permute(0, 2, 1, 3).contiguous().clone()
        d_value_torch = value_torch.grad.permute(0, 2, 1, 3).contiguous().clone()

        return output_torch,  d_query_torch, d_key_torch, d_value_torch


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (128, 128),
        (1024, 1024),
        (128, 256),
        (128, 1024),
        (256, 1024),
        (256, 128),
        (512, 128),
        (768, 128),
        (1024, 128)
    ],
)
@pytest.mark.parametrize("causal", [False, True])
#@pytest.mark.parametrize("causal", [False])
def test_flash_attn_fwd_id_matrix(
        seqlen_q, seqlen_k,  d, causal, dtype
):
    device = "cuda"

    batch_size = 4
    nheads = 4
    query = create_stacked_identity_matrix(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    key = create_stacked_identity_matrix(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    value = create_stacked_identity_matrix(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    d_output = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)

    output, l = fwd(query, key, value, causal)

    output_torch = attention_ref(query=query, key=key, value=value, causal=causal)

    output_max_diff = (output - output_torch).abs().max().item()
    output_mean_diff = (output - output_torch).abs().mean().item()
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

    print(f"output max diff: {output_max_diff}")
    print(f"output mean diff: {output_mean_diff}")



    # if seqlen_q == 1024 and d == 128:
    #     for i in range(32):
    #         print(32*i, output[0,32*i,0,127], output_torch[0,32*i,0,127])

    # with open(f"outputs/test.txt", "w") as f:
    #     f.write("hi test")

    assert output_max_diff <= 1e-2
    assert output_mean_diff <= 1e-4



@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("batch_size", [1,2,3,4])
@pytest.mark.parametrize("nheads", [1,2,3,4])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (128, 128),
        (1024, 1024),
        (128, 256),
        (128, 1024),
        (256, 1024),
        (256, 128),
        (512, 128),
        (768, 128),
        (1024, 128)
    ],
)
@pytest.mark.parametrize("causal", [False, True])
#@pytest.mark.parametrize("causal", [False])
def test_flash_attn_fwd(
        batch_size, nheads, seqlen_q, seqlen_k,  d, causal, dtype
):
    device = "cuda"

    # batch_size = 2
    # nheads = 1
    query = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    key = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    value = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    d_output = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)

    output, l = fwd(query, key, value, causal)

    output_torch = attention_ref(query, key, value, None, causal)

    output_max_diff = (output - output_torch).abs().max().item()
    output_mean_diff = (output - output_torch).abs().mean().item()
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

    print(f"output max diff: {output_max_diff}")
    print(f"output mean diff: {output_mean_diff}")



    # if seqlen_q == 1024 and d == 128:
    #     for i in range(32):
    #         print(32*i, output[0,32*i,0,127], output_torch[0,32*i,0,127])

    #
    # with open(f"outputs/test.txt", "w") as f:
    #     f.write("hi test flash attn")

    assert output_max_diff <= 1e-2
    assert output_mean_diff <= 1e-4



@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (128, 128),
        (1024, 1024),
        (128, 256),
        (128, 1024),
        (256, 1024),
        (256, 128),
        (512, 128),
        (768, 128),
        (1024, 128)
    ],
)
@pytest.mark.parametrize("causal", [False, True])
#@pytest.mark.parametrize("causal", [False])
def test_flash_attn_bwd(
        seqlen_q, seqlen_k,  d, causal, dtype
):
    
    torch.cuda.init()

    device = "cuda"    
    batch_size = 4
    nheads = 4
    query = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    key = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    value = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    d_output = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)

    output, l = fwd(query, key, value, causal)
    d_query, d_key, d_value = bwd(query, key, value, output, l, d_output, causal)

    output_torch,  d_query_torch, d_key_torch, d_value_torch = attention_ref(query, key, value, d_output, causal)
    output_mem_eff_attn, d_query_mem_eff_attn, d_key_mem_eff_attn, d_value_mem_eff_attn = memory_efficient_attention_ref(query, key, value, d_output, causal)


    

    output_max_diff = (output - output_torch).abs().max().item()
    output_mean_diff = (output - output_torch).abs().mean().item()

    dq_max_diff = (d_query - d_query_torch).abs().max().item()
    dk_max_diff = (d_key - d_key_torch).abs().max().item()
    dv_max_diff = (d_value - d_value_torch).abs().max().item()

    dq_mean_diff = (d_query - d_query_torch).abs().mean().item()
    dk_mean_diff = (d_key - d_key_torch).abs().mean().item()
    dv_mean_diff = (d_value - d_value_torch).abs().mean().item()


    output_mea_max_diff = (output - output_mem_eff_attn).abs().max().item()
    output_mea_mean_diff = (output - output_mem_eff_attn).abs().mean().item()

    dq_mea_max_diff = (d_query - d_query_mem_eff_attn).abs().max().item()
    dk_mea_max_diff = (d_key - d_key_mem_eff_attn).abs().max().item()
    dv_mea_max_diff = (d_value - d_value_mem_eff_attn).abs().max().item()

    dq_mea_mean_diff = (d_query - d_query_mem_eff_attn).abs().mean().item()
    dk_mea_mean_diff = (d_key - d_key_mem_eff_attn).abs().mean().item()
    dv_mea_mean_diff = (d_value - d_value_mem_eff_attn).abs().mean().item()

    print("\n")
    print("========================================")
    # print(f"seqlen_q = {seqlen_q}, seqlen_k = {seqlen_k}, d = {d}, causal = {causal}")
    print(f"output max diff: {output_max_diff}")
    print(f"output mean diff: {output_mean_diff}")
    print(f"dQ max diff: {dq_max_diff}")
    print(f"dQ mean diff: {dq_mean_diff}")
    print(f"dK max diff: {dk_max_diff}")
    print(f"dK mean diff: {dk_mean_diff}")
    print(f"dV max diff: {dv_max_diff}")
    print(f"dV mean diff: {dv_mean_diff}")
    print("========================================")


    assert output_max_diff <= 1e-2
    assert output_mean_diff <= 1e-3
    assert dq_max_diff <= 1e-2
    assert dq_mean_diff <= 1e-3
    assert dk_max_diff <= 1e-2
    assert dk_mean_diff <= 1e-3
    assert dv_max_diff <= 1e-2 
    assert dv_mean_diff <= 1e-3


    # assert output_mea_max_diff <= 1e-2
    # assert output_mea_mean_diff <= 1e-3
    # assert dq_mea_max_diff <= 1e-2
    # assert dq_mea_mean_diff <= 1e-3
    # assert dk_mea_max_diff <= 1e-2
    # assert dk_mea_mean_diff <= 1e-3
    # assert dv_mea_max_diff <= 1e-2 
    # assert dv_mea_mean_diff <= 1e-3