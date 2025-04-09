import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
from flash_attn_turing import flash_attn_func, flash_attn_backward_func

torch.set_printoptions(precision=8)


def main():
    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add the 'n'  batch_size=1, seq_len=16, num_heads=1, head_dim=128
    parser.add_argument('batch_size', type=int)
    parser.add_argument('seq_len', type=int)
    parser.add_argument('num_heads', type=int)
    parser.add_argument('head_dim', type=int)

    # Parse the arguments
    args = parser.parse_args()

    # Access the value of n
    batch_size=args.batch_size
    seq_len=args.seq_len
    num_heads=args.num_heads
    head_dim=args.head_dim

    query = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    key = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    value = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    # o = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    # l = torch.randn(batch_size, num_heads, seq_len, dtype=torch.float).to("cuda")

    output, l = flash_attn_func(query, key, value, batch_size, seq_len, num_heads, head_dim)

    d_output = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device="cuda")
    d_q, d_k, d_v = flash_attn_backward_func(query, key, value, output, l, d_output, batch_size, seq_len, num_heads, head_dim)

    print(d_q.size())
    print(d_k.size())
    print(d_v.size())


    query_torch = query.permute(0, 2, 1, 3).contiguous().clone()
    key_torch = key.permute(0, 2, 1, 3).contiguous().clone()
    value_torch = value.permute(0, 2, 1, 3).contiguous().clone()
    d_output_torch = d_output.permute(0, 2, 1, 3).contiguous().clone()

    query_torch.requires_grad = True
    key_torch.requires_grad = True
    value_torch.requires_grad = True

    output = F.scaled_dot_product_attention(query_torch, key_torch, value_torch)


    output.backward(d_output_torch)

    query_torch_grad = query_torch.grad.permute(0, 2, 1, 3).contiguous().clone()
    key_torch_grad = key_torch.grad.permute(0, 2, 1, 3).contiguous().clone()
    value_torch_grad = value_torch.grad.permute(0, 2, 1, 3).contiguous().clone()


    print(query_torch_grad.size())








if __name__ == "__main__":
    main()