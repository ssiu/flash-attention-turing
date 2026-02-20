
import pytest
import torch
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

    batch_size, nheads, seqlen_q, d = query_torch.size()
    _, _, seqlen_k, _ = key_torch.size()

    # compute attention scores
    scores = torch.matmul(query_torch, key_torch.transpose(-2, -1)) / (d ** 0.5)

    if causal:
        attn_mask = causal_lower_right(seqlen_q, seqlen_k, device=scores.device)
        attn_mask = attn_mask.view(1, 1, seqlen_q, seqlen_k)  # broadcast
        scores = scores.masked_fill(~attn_mask, float("-inf"))

    # softmax
    attn = F.softmax(scores, dim=-1)
    attn = torch.where(attn.isnan(), torch.zeros_like(attn), attn)

    # compute output
    output_torch = torch.matmul(attn, value_torch)  # (batch, nheads, seqlen_q, d)
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

    seqlen_q = query_torch.size(2)
    seqlen_k = key_torch.size(2)

    if causal:
        attn_mask = causal_lower_right(seqlen_q, seqlen_k, device=query_torch.device)
    else:
        attn_mask=None

    output_torch = F.scaled_dot_product_attention(query=query_torch, key=key_torch, value=value_torch, attn_mask=attn_mask)

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
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("nheads", [1])
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

    output_diff_np = (output - output_torch).detach().cpu().numpy()
    output_np = output.detach().cpu().numpy()
    output_torch_np = output_torch.detach().cpu().numpy()
    query_np = query.detach().cpu().numpy()
    
    output_diff_np = output_diff_np.squeeze(0).squeeze(1)  # shape (s, d)
    output_np = output_np.squeeze(0).squeeze(1)
    output_torch_np = output_torch_np.squeeze(0).squeeze(1)
    query_np = query_np.squeeze(0).squeeze(1)


    l_np = l.detach().cpu().numpy()   # shape (1, 1, seqlen_q)

    # Squeeze to (seqlen_q,)
    l_np = l_np.squeeze(0).squeeze(0)

    l_df = pd.DataFrame({
        "seqlen": np.arange(seqlen_q),
        "l": l_np
    })

    # Create DataFrame with coordinates
    s_indices, d_indices = torch.meshgrid(torch.arange(seqlen_q), torch.arange(d), indexing='ij')
    df = pd.DataFrame({
        'seqlen': s_indices.flatten().numpy(),
        'd': d_indices.flatten().numpy(),
        'diff': output_diff_np.flatten(),
        'output': output_np.flatten(),
        'output_torch': output_torch_np.flatten(),
        'query': query_np.flatten()
    })

    now = datetime.now()
    excel_path = f"/outputs/{now}_fwd_seqlen_q_{seqlen_q}_seqlen_k_{seqlen_k}_hdim_{d}_causal_{causal}.xlsx"

    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name="output_diff", index=False)
        l_df.to_excel(writer, sheet_name="l_values", index=False)


    assert output_max_diff <= 1e-2
    assert output_mean_diff <= 1e-3

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
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("nheads", [16])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [True])
# @pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
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
        (8192,8292),
    ],
)
def test_flash_attn_bwd(        
    batch_size, nheads, seqlen_q, seqlen_k,  d, causal, dtype
):
    
    torch.cuda.init()
    torch.cuda.current_device()
    device = "cuda"    
    query = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    key = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    value = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
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

    output_max_diff = (output - output_torch).abs().max().item()
    output_mean_diff = (output - output_torch).abs().mean().item()

    dq_max_diff = (d_query - d_query_torch).abs().max().item()
    dk_max_diff = (d_key - d_key_torch).abs().max().item()
    dv_max_diff = (d_value - d_value_torch).abs().max().item()

    dq_mean_diff = (d_query - d_query_torch).abs().mean().item()
    dk_mean_diff = (d_key - d_key_torch).abs().mean().item()
    dv_mean_diff = (d_value - d_value_torch).abs().mean().item()



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


    # def get_cpu_tensors(t, t_torch=None):
    #     if t_torch is None:
    #         t_np = t.detach().cpu().numpy().squeeze(0).squeeze(1)
    #         return t_np
    #     else:
    #         t_diff_np = (t-t_torch).detach().cpu().numpy().squeeze(0).squeeze(1)
    #         t_np = t.detach().cpu().numpy().squeeze(0).squeeze(1)
    #         t_torch_np = t_torch.detach().cpu().numpy().squeeze(0).squeeze(1)

    #         return t_diff_np, t_np, t_torch_np


    # query_np = get_cpu_tensors(query)
    # key_np = get_cpu_tensors(key)
    # value_np = get_cpu_tensors(value)

    # output_diff_np, output_np, output_torch_np = get_cpu_tensors(output, output_torch)
    # d_query_diff_np, d_query_np, d_query_torch_np = get_cpu_tensors(d_query, d_query_torch)
    # d_key_diff_np, d_key_np, d_key_torch_np = get_cpu_tensors(d_key, d_key_torch)
    # d_value_diff_np, d_value_np, d_value_torch_np = get_cpu_tensors(d_value, d_value_torch)




    # l_np = l.detach().cpu().numpy()   # shape (1, 1, seqlen_q)

    # # Squeeze to (seqlen_q,)
    # l_np = l_np.squeeze(0).squeeze(0)

    # l_df = pd.DataFrame({
    #     "seqlen": np.arange(seqlen_q),
    #     "l": l_np
    # })

    # s_indices, d_indices = torch.meshgrid(torch.arange(seqlen_q), torch.arange(d), indexing='ij')
    # df = pd.DataFrame({
    #     'seqlen': s_indices.flatten().numpy(),
    #     'd': d_indices.flatten().numpy(),
    #     'output_diff': output_diff_np.flatten(),
    #     'output': output_np.flatten(),
    #     'output_torch': output_torch_np.flatten(),
    #     'd_query_diff': d_query_diff_np.flatten(),
    #     'd_query': d_query_np.flatten(),
    #     'd_query_torch': d_query_torch_np.flatten(),
    #     'query': query_np.flatten()
    # })

    # s_indices, d_indices = torch.meshgrid(torch.arange(seqlen_k), torch.arange(d), indexing='ij')
    # df_kv = pd.DataFrame({
    #     'seqlen': s_indices.flatten().numpy(),
    #     'd': d_indices.flatten().numpy(),
    #     'd_key_diff': d_key_diff_np.flatten(),
    #     'd_key': d_key_np.flatten(),
    #     'd_key_torch': d_key_torch_np.flatten(),
    #     'd_value_diff': d_value_diff_np.flatten(),
    #     'd_value': d_value_np.flatten(),
    #     'd_value_torch': d_value_torch_np.flatten(),
    #     'key': key_np.flatten(),
    #     'value': value_np.flatten()
    # })

    # now = datetime.now()

    # # df.to_excel(f"/outputs/{now}_bwd_seqlen_q_{seqlen_q}_seqlen_k_{seqlen_k}_hdim_{d}_causal_{causal}.xlsx")
    # excel_path = f"/outputs/{now}_bwd_seqlen_q_{seqlen_q}_seqlen_k_{seqlen_k}_hdim_{d}_causal_{causal}.xlsx"
    # with pd.ExcelWriter(excel_path) as writer:
    #     df.to_excel(writer, sheet_name="output_diff", index=False)
    #     l_df.to_excel(writer, sheet_name="l_values", index=False)
    #     df_kv.to_excel(writer, sheet_name="kv_values", index=False)

    assert output_max_diff <= 2e-2
    assert output_mean_diff <= 2e-3
    assert dq_max_diff <= 2e-2
    assert dq_mean_diff <= 2e-3
    assert dk_max_diff <= 2e-2
    assert dk_mean_diff <= 2e-3
    assert dv_max_diff <= 2e-2
    assert dv_mean_diff <= 2e-3
