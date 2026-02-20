import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

from flash_attn_turing import (
    fwd,
    bwd
)
import argparse

torch.set_printoptions(precision=8)

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


def get_error(batch_size=1, seq_len=16, num_heads=1, head_dim=128, is_causal=False):
    print(f"Computing error for batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")
    query = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    key = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    value = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    d_output = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    
    # query = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    # key = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    # value = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")

    output, l = fwd(query, key, value, is_causal)

    d_query, d_key, d_value = bwd(query, key, value, output, l, d_output, is_causal)
    # for pytorch function
    # (batch_size, num_heads, seq_len, head_dim)


    # (batch_size, num_heads, seq_len, head_dim)
    if not is_causal:
        query_torch = query.permute(0, 2, 1, 3).contiguous().clone().requires_grad_(True)
        key_torch = key.permute(0, 2, 1, 3).contiguous().clone().requires_grad_(True)
        value_torch = value.permute(0, 2, 1, 3).contiguous().clone().requires_grad_(True)
        d_output_torch = d_output.permute(0, 2, 1, 3).contiguous().clone()

        output_torch = F.scaled_dot_product_attention(query_torch, key_torch, value_torch)
        d_query_torch, d_key_torch, d_value_torch = torch.autograd.grad(
            outputs=output_torch,
            inputs=(query_torch, key_torch, value_torch),
            grad_outputs=d_output_torch,
            retain_graph=False,
            allow_unused=False,
        )

        output_torch = output_torch.permute(0, 2, 1, 3).contiguous().clone()
        d_query_torch = d_query_torch.permute(0, 2, 1, 3).contiguous().clone()
        d_key_torch = d_key_torch.permute(0, 2, 1, 3).contiguous().clone()
        d_value_torch = d_value_torch.permute(0, 2, 1, 3).contiguous().clone()

    else:
        output_torch,  d_query_torch, d_key_torch, d_value_torch = attention_ref(query, key, value, d_output, is_causal)





    sum_error = torch.sum(torch.abs(output - output_torch))
    avg_error = sum_error / (batch_size * seq_len * num_heads * head_dim)
    max_error = torch.max(torch.abs(output - output_torch))

    max_error_index = torch.argmax(torch.abs(output - output_torch))

    # Convert the flat index to multi-dimensional indices (if needed)
    max_error_indices = torch.unravel_index(max_error_index, output.shape)

    # Extract the values at the maximum error location
    output_value = output[max_error_indices]
    output_torch_value = output_torch[max_error_indices]


    return sum_error, avg_error, max_error, output_value, output_torch_value



def main():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        val = v.lower()
        if val in ("true", "1", "t", "yes", "y"):
            return True
        if val in ("false", "0", "f", "no", "n"):
            return False
        raise argparse.ArgumentTypeError("is_causal must be true/false")

    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add the 'n'  batch_size=1, seq_len=16, num_heads=1, head_dim=128
    parser.add_argument('batch_size', type=int)
    parser.add_argument('seq_len', type=int)
    parser.add_argument('num_heads', type=int)
    parser.add_argument('head_dim', type=int)
    parser.add_argument('is_causal', type=str2bool)

    # Parse the arguments
    args = parser.parse_args()

    # Access the value of n
    batch_size=args.batch_size
    seq_len=args.seq_len
    num_heads=args.num_heads
    head_dim=args.head_dim
    is_causal=args.is_causal

    # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    sum_error, avg_error, max_error, output_value, output_torch_value = get_error(
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        is_causal=is_causal,
    )

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(f"seqlen = {seq_len}, head_dim = {head_dim}, is_causal = {is_causal}, sum_error = {sum_error}, avg_error = {avg_error}, max_error = {max_error},\nmax_error output = {output_value}, max_error output torch = {output_torch_value}")



if __name__ == "__main__":
    main()
