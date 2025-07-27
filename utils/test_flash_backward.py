import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
from flash_attn_turing import fwd, bwd

torch.set_printoptions(precision=8)
torch.set_printoptions(linewidth=500)

def get_error(tensor, tensor_torch, batch_size, seq_len, num_heads, head_dim):
    sum_error = torch.sum(torch.abs(tensor - tensor_torch))
    avg_error = sum_error / (batch_size * seq_len * num_heads * head_dim)
    max_error = torch.max(torch.abs(tensor - tensor_torch))

    max_error_index = torch.argmax(torch.abs(tensor - tensor_torch))

    # Convert the flat index to multi-dimensional indices (if needed)
    max_error_indices = torch.unravel_index(max_error_index, tensor.shape)

    # Extract the values at the maximum error location
    output_value = tensor[max_error_indices]
    output_torch_value = tensor_torch[max_error_indices]


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

    # query = torch.zeros(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    # key = torch.zeros(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    # value = torch.zeros(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")

    # query = torch.ones(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    # key = torch.ones(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    # value = torch.ones(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")


    # device = torch.device('cuda')
    # query = torch.eye(128, device=device, dtype=torch.float16).view(1, 128, 1, 128)
    # key = torch.eye(128, device=device, dtype=torch.float16).view(1, 128, 1, 128)
    # value = torch.eye(128, device=device, dtype=torch.float16).view(1, 128, 1, 128)
    # d_output = torch.eye(128, device=device, dtype=torch.float16).view(1, 128, 1, 128)

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
    # d_output = concat_tensor.clone()


    # 128 x 64
    #
    # eye1 = torch.eye(64, dtype=torch.float16)
    #
    # # Create the second identity matrix
    # eye2 = torch.eye(64, dtype=torch.float16)
    #
    # # Concatenate them vertically using torch.cat
    # concat_matrix = torch.cat((eye1, eye2), dim=0)
    #
    # concat_tensor = concat_matrix.view(1,128,1,64).to("cuda")
    #
    # query = concat_tensor.clone()
    # key = concat_tensor.clone()
    # value = concat_tensor.clone()
    # d_output = concat_tensor.clone()


    query = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    key = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    value = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    d_output = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device="cuda")

    # o = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).to("cuda")
    # l = torch.randn(batch_size, num_heads, seq_len, dtype=torch.float).to("cuda")
    #is_causal = 1
    output, l = fwd(query, key, value, batch_size, seq_len, num_heads, head_dim, bool(is_causal))

    # print("values of l")
    #
    # for i in range(128):
    #     print(l[0,0,i])
    #
    # print("=====")



    #d_output = torch.ones(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16, device="cuda")





    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        d_query, d_key, d_value = bwd(query, key, value, output, l, d_output, batch_size, seq_len, num_heads, head_dim, bool(is_causal))

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


    print(d_query.size())


    query_torch = query.permute(0, 2, 1, 3).contiguous().clone()
    key_torch = key.permute(0, 2, 1, 3).contiguous().clone()
    value_torch = value.permute(0, 2, 1, 3).contiguous().clone()
    d_output_torch = d_output.permute(0, 2, 1, 3).contiguous().clone()

    query_torch.requires_grad = True
    key_torch.requires_grad = True
    value_torch.requires_grad = True

    output_torch = F.scaled_dot_product_attention(query=query_torch, key=key_torch, value=value_torch, is_causal=bool(is_causal))


    output_torch.backward(d_output_torch)
    output_torch = output_torch.permute(0, 2, 1, 3).contiguous().clone()

    d_query_torch = query_torch.grad.permute(0, 2, 1, 3).contiguous().clone()
    d_key_torch = key_torch.grad.permute(0, 2, 1, 3).contiguous().clone()
    d_value_torch = value_torch.grad.permute(0, 2, 1, 3).contiguous().clone()


    print("Comparing O")
    sum_error, avg_error, max_error, output_value, output_torch_value = \
        get_error(output, output_torch, batch_size, seq_len, num_heads, head_dim)

    print(f"sum_error = {sum_error}, avg_error = {avg_error}, max_error = {max_error},\nmax_error output = {output_value}, max_error output torch = {output_torch_value}")


    print("==========")


    print("Comparing dV")

    sum_error, avg_error, max_error, output_value, output_torch_value = \
        get_error(d_value, d_value_torch, batch_size, seq_len, num_heads, head_dim)

    print(f"sum_error = {sum_error}, avg_error = {avg_error}, max_error = {max_error},\nmax_error output = {output_value}, max_error output torch = {output_torch_value}")


    print("==========")

    print("Comparing dK")

    sum_error, avg_error, max_error, output_value, output_torch_value = \
        get_error(d_key, d_key_torch, batch_size, seq_len, num_heads, head_dim)

    print(f"sum_error = {sum_error}, avg_error = {avg_error}, max_error = {max_error},\nmax_error output = {output_value}, max_error output torch = {output_torch_value}")


    print("==========")

    print("Comparing dQ")

    sum_error, avg_error, max_error, output_value, output_torch_value = \
        get_error(d_query, d_query_torch, batch_size, seq_len, num_heads, head_dim)

    print(f"sum_error = {sum_error}, avg_error = {avg_error}, max_error = {max_error},\nmax_error output = {output_value}, max_error output torch = {output_torch_value}")

    print("All done! YEAH")

    print("==========")

    # for i in range(32):
    #     row = d_key[:, i, :, :]  # shape: (1, 1, 128)
    #     row_torch = d_key_torch[:, i, :, :]
    #     print(f"dk, i = {i}");
    #     print(f"{row}\n")
    #     print(f"{row_torch}\n")
    #     print("==================================================")

    # i = 191 HAS PROBLEMS
######################### 12/6/2025 debug

    # diff = d_query - d_query_torch
    # # #diff = d_key - d_key_torch
    # #
    # for i in range(32):
    #     row = d_query[:, i, :, :]  # shape: (1, 1, 128)
    #     row_torch = d_query_torch[:, i, :, :]
    #     print(f"dv, i = {i}");
    #     print(f"{row}\n")
    #     print(f"{row_torch}\n")
    #
    #     row = diff[:, i, :, :]
    #     print(f"diff, i = {i}");
    #     print(f"{row}\n")
    #
    #     print("==================================================")
    #
    # for i in range(64, 96):
    #     row = d_query[:, i, :, :]  # shape: (1, 1, 128)
    #     row_torch = d_query_torch[:, i, :, :]
    #     print(f"dv, i = {i}");
    #     print(f"{row}\n")
    #     print(f"{row_torch}\n")
    #
    #     row = diff[:, i, :, :]
    #     print(f"diff, i = {i}");
    #     print(f"{row}\n")
    #
    #     print("==================================================")
######################### 12/6/2025 debug

    # for i in range(128):
    #     for j in range(64):
    #         if diff[:, i, :, j] > 0.0001:
    #             print(i,j)

    #
    # for i in range(128):
    #     print(f"i = {i}, dQ = {d_query[0,0,0,i]}, dQ_torch = {d_query_torch[0,0,0,i]}")
    #
    # print("##################################################")
    # for i in range(128):
    #     print(f"i = {i}, dQ = {d_query[0,i,0,0]}, dQ_torch = {d_query_torch[0,i,0,0]}")
    #
    #DO NOT UNCOMMENT, ONLY SHOWS l
    # print("##################################################")
    # for i in range(64):
    #     print(f"i = {i}, l = {l[0,0,i]}")

######################### 12/6/2025 debug
    # print(f"o = {output[0,0,0,:]}")
    # print(f"do = {d_output[0,0,0,:]}")
######################### 12/6/2025 debug
if __name__ == "__main__":
    main()