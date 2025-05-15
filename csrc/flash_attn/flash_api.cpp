#include <torch/extension.h>
#include "flash.h"
#include "static_switch.h"
//std::vector<torch::Tensor> flash_fwd(torch::Tensor q,
//                                     torch::Tensor k,
//                                     torch::Tensor v,
//                                     int batch_size,
//                                     int seq_len,
//                                     int num_heads,
//                                     int head_dim);
//
//
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

void run_mha_fwd(half_t* q,
                 half_t* k,
                 half_t* v,
                 half_t* o,
                 float* l,
                 int batch_size, int seq_len, int num_heads, int head_dim, int is_causal){
    HEADDIM_SWITCH(head_dim, [&] {
        BOOL_SWITCH(is_causal, Is_causal, [&] {
                run_mha_fwd_<kHeadDim, Is_causal>(q, k, v, o, l,
                        batch_size, seq_len, num_heads, head_dim, is_causal);
        });
    });
}


std::vector<torch::Tensor>
mha_fwd(torch::Tensor q,
             torch::Tensor k,
             torch::Tensor v,
             int batch_size,
             int seq_len,
             int num_heads,
             int head_dim,
             int is_causal)
{
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

    run_mha_fwd(q_ptr,
                k_ptr,
                v_ptr,
                o_ptr,
                l_ptr,
                batch_size, seq_len, num_heads, head_dim, is_causal);


    return {o, l};

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_fwd_func", &mha_fwd, "Forward pass");
    m.def("flash_attn_bwd_func", &flash_bwd, "Backward pass");
}