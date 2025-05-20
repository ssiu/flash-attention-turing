import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from flash_attn_turing import flash_attn_fwd_func
import argparse

torch.set_printoptions(precision=8)


def get_error(batch_size=1, seq_len=16, num_heads=1, head_dim=128, is_causal=1):
    print(f"Computing error for batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")
    # query = torch.zeros(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    # key = torch.zeros(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    # value = torch.zeros(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")

    # device = torch.device('cuda')
    # query = torch.eye(128, dtype=torch.float16).view(1, 128, 1, 128).to("cuda")
    # key = torch.eye(128, dtype=torch.float16).view(1, 128, 1, 128).to("cuda")
    # value = torch.eye(128, dtype=torch.float16).view(1, 128, 1, 128).to("cuda")

    # 256 x 128
    # eye1 = torch.eye(128, dtype=torch.float16)
    #
    # # Create the second identity matrix
    # eye2 = torch.eye(128, dtype=torch.float16)
    #
    # # Concatenate them vertically using torch.cat
    # concat_matrix = torch.cat((eye1, eye2), dim=0)
    #
    # concat_tensor = concat_matrix.view(1,256,1,128).to("cuda")
    #
    # query = concat_tensor.clone()
    # key = concat_tensor.clone()
    # value = concat_tensor.clone()

    query = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    key = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    value = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")

    # for pytorch function
    # (batch_size, num_heads, seq_len, head_dim)
    query_torch = query.permute(0, 2, 1, 3).contiguous().clone()
    key_torch = key.permute(0, 2, 1, 3).contiguous().clone()
    value_torch = value.permute(0, 2, 1, 3).contiguous().clone()


    output, l = flash_attn_fwd_func(query, key, value, batch_size, seq_len, num_heads, head_dim, is_causal)

    print(l.shape)
    #print(l[:128])
    # (batch_size, num_heads, seq_len, head_dim)
    output_torch = F.scaled_dot_product_attention(query=query_torch, key=key_torch, value=value_torch, is_causal=bool(is_causal))

    output_torch = output_torch.permute(0, 2, 1, 3).contiguous().clone()

    sum_error = torch.sum(torch.abs(output - output_torch))
    avg_error = sum_error / (batch_size * seq_len * num_heads * head_dim)
    max_error = torch.max(torch.abs(output - output_torch))

    max_error_index = torch.argmax(torch.abs(output - output_torch))

    # Convert the flat index to multi-dimensional indices (if needed)
    max_error_indices = torch.unravel_index(max_error_index, output.shape)

    # Extract the values at the maximum error location
    output_value = output[max_error_indices]
    output_torch_value = output_torch[max_error_indices]

    # for i in range(32):
    #     row = output[:, i, :, :]  # shape: (1, 1, 128)
    #     row_torch = output_torch[:, i, :, :]
    #     print(f"i = {i}");
    #     print(f"{row}\n")
    #     print(f"{row_torch}\n")
    #     print("==================================================")

    # for i in range(128, 128+32):
    #     row = output[:, i, :, :]  # shape: (1, 1, 128)
    #     row_torch = output_torch[:, i, :, :]
    #     print(f"i = {i}");
    #     print(f"{row}\n")
    #     print(f"{row_torch}\n")
    #     print("==================================================")


    return sum_error, avg_error, max_error, output_value, output_torch_value



def main():
    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add the 'n'  batch_size=1, seq_len=16, num_heads=1, head_dim=128
    parser.add_argument('batch_size', type=int)
    parser.add_argument('seq_len', type=int)
    parser.add_argument('num_heads', type=int)
    parser.add_argument('head_dim', type=int)
    parser.add_argument('is_causal', type=int)
    # Parse the arguments
    args = parser.parse_args()

    # Access the value of n
    batch_size=args.batch_size
    seq_len=args.seq_len
    num_heads=args.num_heads
    head_dim=args.head_dim
    is_causal=args.is_causal

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        sum_error, avg_error, max_error, output_value, output_torch_value = get_error(batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim, is_causal=is_causal)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(f"sum_error = {sum_error}, avg_error = {avg_error}, max_error = {max_error},\nmax_error output = {output_value}, max_error output torch = {output_torch_value}")



if __name__ == "__main__":
    main()