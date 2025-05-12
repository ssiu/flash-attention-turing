#include <torch/extension.h>


//std::vector<torch::Tensor> flash_fwd(torch::Tensor q,
//                                     torch::Tensor k,
//                                     torch::Tensor v,
//                                     int batch_size,
//                                     int seq_len,
//                                     int num_heads,
//                                     int head_dim);


std::vector<torch::Tensor>
flash_fwd(torch::Tensor q,
             torch::Tensor k,
             torch::Tensor v,
             int batch_size, int seq_len, int num_heads, int head_dim)
{
 constexpr int kBlockM = 128;
 constexpr int kBlockN = 64;
 constexpr int kHeadDim = 128;

 auto device = q.device();

 torch::Tensor o = torch::zeros(q.sizes(), q.options().dtype(torch::kFloat16));

 std::vector<int64_t> size = {batch_size, num_heads, seq_len};
 torch::Tensor l = torch::empty(size, q.options().dtype(torch::kFloat32).device(device));

 TORCH_CHECK(o.is_cuda(), "Tensor o is not on CUDA");

 half_t* q_ptr = reinterpret_cast<half_t*>(q.data_ptr());
 half_t* k_ptr = reinterpret_cast<half_t*>(k.data_ptr());
 half_t* v_ptr = reinterpret_cast<half_t*>(v.data_ptr());
 half_t* o_ptr = reinterpret_cast<half_t*>(o.data_ptr());

 float* l_ptr = reinterpret_cast<float*>(l.data_ptr());

 dim3 dimGrid(batch_size, num_heads, seq_len / kBlockM);
 dim3 dimBlock(256);
 int maxbytes = 65536;



 auto kernel = flash_fwd_kernel<Flash_fwd_kernel_traits<kHeadDim, kBlockM, kBlockN, 8>>;
 cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);


 kernel<<<dimGrid, dimBlock, maxbytes>>>(q_ptr,
                                         k_ptr,
                                         v_ptr,
                                         o_ptr,
                                         l_ptr,
                                         batch_size, seq_len, num_heads, head_dim);
 return {o, l};
}


std::vector<torch::Tensor> flash_bwd(torch::Tensor q,
                                     torch::Tensor k,
                                     torch::Tensor v,
                                     torch::Tensor o,
                                     torch::Tensor l,
                                     torch::Tensor d_o,
                                     int batch_size,
                                     int seq_len,
                                     int num_heads,
                                     int head_dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_func", &flash_fwd, "flash_fwd");
    m.def("flash_attn_backward_func", &flash_bwd, "flash_bwd");
}