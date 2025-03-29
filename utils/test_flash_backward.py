import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import argparse

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


    #batch_size, num_heads, seq_len, head_dim = 4, 32, 4096, 128
    Q = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    K = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    V = torch.rand(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    dO = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    # with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
    #     output =  F.scaled_dot_product_attention(query, key, value)
    output = F.scaled_dot_product_attention(Q, K, V)
    output.backward(dO)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))



if __name__ == "__main__":
    main()