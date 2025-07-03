#include <torch/extension.h>
#include "flash.h"
#include "static_switch.h"


void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *softmax_lse_d,
                      float softmax_scale,
                      int is_causal) {

    // Reset the parameters
    params = {};

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);
    }

//    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
//    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
//    params.seqused_k = static_cast<int *>(seqused_k);


    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;


    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    params.is_causal = is_causal;
}




void run_mha_fwd(Flash_fwd_params &params){
    HEADDIM_SWITCH(params.d, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_fwd_<kHeadDim, Is_causal>(params);
        });
    });
}

void run_mha_bwd(half_t* q,
                 half_t* k,
                 half_t* v,
                 half_t* o,
                 float* l,
                 float* d,
                 half_t* do_,
                 float* dq_float,
                 half_t* dq,
                 half_t* dk,
                 half_t* dv,
                 int batch_size, int seq_len, int num_heads, int head_dim, int is_causal){
    HEADDIM_SWITCH(head_dim, [&] {
        BOOL_SWITCH(is_causal, Is_causal, [&] {
                run_mha_bwd_<kHeadDim, Is_causal>(q, k, v, o, l, d, do_, dq_float, dq, dk, dv,
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

//    half_t* q_ptr = reinterpret_cast<half_t*>(q.data_ptr());
//    half_t* k_ptr = reinterpret_cast<half_t*>(k.data_ptr());
//    half_t* v_ptr = reinterpret_cast<half_t*>(v.data_ptr());
//    half_t* o_ptr = reinterpret_cast<half_t*>(o.data_ptr());
//
//    float* l_ptr = reinterpret_cast<float*>(l.data_ptr());

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size,
                     q, k, v, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     l.data_ptr(),
                     softmax_scale,
                     is_causal
                     );

    run_mha_fwd(params);


    return {o, l};

}

std::vector<torch::Tensor>
mha_bwd(torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor o,
        torch::Tensor l,
        torch::Tensor do_,
        int batch_size,
        int seq_len,
        int num_heads,
        int head_dim,
        int is_causal)
{

    torch::Tensor dq_float = torch::zeros(q.sizes(), q.options().dtype(torch::kFloat32));
    torch::Tensor dq = torch::zeros(q.sizes(), q.options().dtype(torch::kFloat16));
    torch::Tensor dk = torch::empty(k.sizes(), k.options().dtype(torch::kFloat16));
    torch::Tensor dv = torch::empty(v.sizes(), v.options().dtype(torch::kFloat16));

    torch::Tensor d = torch::empty(l.sizes(), l.options());

    half_t* q_ptr = reinterpret_cast<half_t*>(q.data_ptr());
    half_t* k_ptr = reinterpret_cast<half_t*>(k.data_ptr());
    half_t* v_ptr = reinterpret_cast<half_t*>(v.data_ptr());
    half_t* o_ptr = reinterpret_cast<half_t*>(o.data_ptr());
    float* l_ptr = reinterpret_cast<float*>(l.data_ptr());
    float* d_ptr = reinterpret_cast<float*>(d.data_ptr());
    half_t* do_ptr = reinterpret_cast<half_t*>(do_.data_ptr());

    float* dq_float_ptr = reinterpret_cast<float*>(dq_float.data_ptr());
    half_t* dq_ptr = reinterpret_cast<half_t*>(dq.data_ptr());
    half_t* dk_ptr = reinterpret_cast<half_t*>(dk.data_ptr());
    half_t* dv_ptr = reinterpret_cast<half_t*>(dv.data_ptr());

    run_mha_bwd(q_ptr,
                k_ptr,
                v_ptr,
                o_ptr,
                l_ptr,
                d_ptr,
                do_ptr,
                dq_float_ptr,
                dq_ptr,
                dk_ptr,
                dv_ptr,
                batch_size, seq_len, num_heads, head_dim, is_causal);


    return {dq, dk, dv};

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_fwd_func", &mha_fwd, "Forward pass");
    m.def("flash_attn_bwd_func", &mha_bwd, "Backward pass");
}