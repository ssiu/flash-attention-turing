#include <torch/extension.h>


std::vector<torch::Tensor> flash_fwd(torch::Tensor q,
                                     torch::Tensor k,
                                     torch::Tensor v,
                                     int batch_size,
                                     int seq_len,
                                     int num_heads,
                                     int head_dim);
//
//std::vector<torch::Tensor> flash_bwd_v1(torch::Tensor q,
//                                     torch::Tensor k,
//                                     torch::Tensor v,
//                                     torch::Tensor o,
//                                     torch::Tensor l,
//                                     torch::Tensor d_o,
//                                     int batch_size,
//                                     int seq_len,
//                                     int num_heads,
//                                     int head_dim);


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