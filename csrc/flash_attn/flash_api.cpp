#include <torch/extension.h>


torch::Tensor flash_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                            int batch_size, int seq_len, int num_heads, int head_dim);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_func", &flash_fwd, "flash_fwd");
}