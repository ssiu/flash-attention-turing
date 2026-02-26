#include <torch/extension.h>
#include "flash.h"
#include "static_switch.h"

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      at::Tensor l,
                      //void *softmax_lse_d,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      bool is_causal) {

    // Reset the parameters
    params = {};

    // Set the pointers and strides.
    params.q_ptr = reinterpret_cast<half_t*>(q.data_ptr());
    params.k_ptr = reinterpret_cast<half_t*>(k.data_ptr());
    params.v_ptr = reinterpret_cast<half_t*>(v.data_ptr());
    params.o_ptr = reinterpret_cast<half_t*>(out.data_ptr());


    // Softmax sum
    params.l_ptr = reinterpret_cast<float*>(l.data_ptr());





    // Set the dimensions.
    params.b = b;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.d = d;
    params.is_causal = is_causal;
    params.cu_seqlens_q_ptr = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k_ptr = static_cast<int *>(cu_seqlens_k_d);
    params.is_seqlens_k_cumulative = true;
}


void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor out,
                      const at::Tensor l,
                      const at::Tensor dout,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      at::Tensor do_o,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      //void *softmax_lse_d,
                      bool is_causal) {


    set_params_fprop(params,
                     b,
                     seqlen_q,
                     seqlen_k,
                     h,
                     h_k,
                     d,
                     q, k, v, out, l,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     is_causal
                     );


    params.do_o_ptr = reinterpret_cast<float*>(do_o.data_ptr());
    params.do_ptr = reinterpret_cast<half_t*>(dout.data_ptr());

    params.dq_ptr = reinterpret_cast<half_t*>(dq.data_ptr());
    params.dk_ptr = reinterpret_cast<half_t*>(dk.data_ptr());
    params.dv_ptr = reinterpret_cast<half_t*>(dv.data_ptr());





}


void run_mha_fwd(Flash_fwd_params &params){
    HEADDIM_SWITCH(params.d, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_fwd_<kHeadDim, Is_causal>(params);
        });
    });
}

void run_mha_bwd(Flash_bwd_params &params){
    HEADDIM_SWITCH(params.d, [&] {
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_bwd_<kHeadDim, Is_causal>(params);
        });
    });
}


std::vector<torch::Tensor>
mha_fwd(torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
//             int batch_size,
//             int seq_len,
//             int num_heads,
//             int head_dim,
        bool is_causal)
{
    auto device = q.device();

    const auto sizes = q.sizes();

    int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    int head_size = sizes[3];

    int seqlen_k = k.size(1);
    int num_heads_k = k.size(2);

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be rank-4 tensors");
    TORCH_CHECK(k.size(0) == batch_size && v.size(0) == batch_size, "k/v batch size must match q");
    TORCH_CHECK(v.size(1) == seqlen_k, "k and v seqlen_k must match");
    TORCH_CHECK(v.size(2) == num_heads_k, "k and v num_heads must match");
    TORCH_CHECK(k.size(3) == head_size && v.size(3) == head_size, "q/k/v head_dim must match");
    TORCH_CHECK(num_heads % num_heads_k == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");

    torch::Tensor o = torch::zeros(q.sizes(), q.options().dtype(torch::kFloat16));

    std::vector<int64_t> size = {batch_size, num_heads, seqlen_q};
    torch::Tensor l = torch::zeros(size, q.options().dtype(torch::kFloat32).device(device));

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
                     seqlen_q,
                     seqlen_k,
                     num_heads,
                     num_heads_k,
                     head_size,
                     q, k, v, o, l,
                     nullptr,
                     nullptr,
                     is_causal
                     );

    // std::cout << "Q ptr: " << q.data_ptr() << "\n";
    // std::cout << "K ptr: " << k.data_ptr() << "\n";
    // std::cout << "V ptr: " << v.data_ptr() << "\n";
    // std::cout << "O ptr: " << o.data_ptr() << "\n";

    run_mha_fwd(params);


    return {o, l};

}




std::vector<torch::Tensor>
mha_bwd(torch::Tensor q,
        torch::Tensor k,
        torch::Tensor v,
        torch::Tensor out,
        torch::Tensor l,
        torch::Tensor dout,
//        int batch_size,
//        int seq_len,
//        int num_heads,
//        int head_dim,
        bool is_causal)
{

    const auto sizes = q.sizes();

    int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    int head_size = sizes[3];

    int seqlen_k = k.size(1);
    int num_heads_k = k.size(2);

    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q, k, v must be rank-4 tensors");
    TORCH_CHECK(out.dim() == 4 && dout.dim() == 4, "out and dout must be rank-4 tensors");
    TORCH_CHECK(k.size(0) == batch_size && v.size(0) == batch_size, "k/v batch size must match q");
    TORCH_CHECK(v.size(1) == seqlen_k, "k and v seqlen_k must match");
    TORCH_CHECK(v.size(2) == num_heads_k, "k and v num_heads must match");
    TORCH_CHECK(k.size(3) == head_size && v.size(3) == head_size, "q/k/v head_dim must match");
    TORCH_CHECK(out.sizes() == q.sizes() && dout.sizes() == q.sizes(), "out and dout must match q shape");
    TORCH_CHECK(num_heads % num_heads_k == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");

    torch::Tensor dq = torch::zeros(q.sizes(), q.options().dtype(torch::kFloat16));
    torch::Tensor dk = torch::zeros(k.sizes(), k.options().dtype(torch::kFloat16));
    torch::Tensor dv = torch::zeros(v.sizes(), v.options().dtype(torch::kFloat16));
    torch::Tensor dk_expanded = dk;
    torch::Tensor dv_expanded = dv;
    if (num_heads != num_heads_k) {
        dk_expanded = torch::zeros({batch_size, seqlen_k, num_heads, head_size}, k.options().dtype(torch::kFloat16));
        dv_expanded = torch::zeros({batch_size, seqlen_k, num_heads, head_size}, v.options().dtype(torch::kFloat16));
    }

    torch::Tensor do_o = torch::zeros(l.sizes(), l.options());

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                    seqlen_q,
                    seqlen_k,
                    num_heads,
                    num_heads_k,
                    head_size,
                    q,
                    k,
                    v,
                    out,
                    l,
                    dout,
                    dq,
                    dk_expanded,
                    dv_expanded,
                    do_o,
                    nullptr,
                    nullptr,
                    is_causal);

    run_mha_bwd(params);

    if (num_heads != num_heads_k) {
        torch::sum_out(
            dk,
            torch::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}),
            {3}
        );
        torch::sum_out(
            dv,
            torch::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}),
            {3}
        );
    }


    return {dq, dk, dv};

}

std::vector<torch::Tensor>
mha_varlen_fwd(torch::Tensor q,
               torch::Tensor k,
               torch::Tensor v,
               torch::Tensor cu_seqlens_q,
               torch::Tensor cu_seqlens_k,
               int64_t max_seqlen_q,
               int64_t max_seqlen_k,
               bool is_causal)
{
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q, k, v must be rank-3 packed tensors");
    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(), "cu_seqlens_q/cu_seqlens_k must be CUDA tensors");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::kInt32 && cu_seqlens_k.scalar_type() == torch::kInt32,
                "cu_seqlens_q/cu_seqlens_k must be int32 tensors");
    TORCH_CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(),
                "cu_seqlens_q/cu_seqlens_k must be contiguous");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "cu_seqlens_q/cu_seqlens_k must be rank-1");
    TORCH_CHECK(cu_seqlens_q.numel() >= 2 && cu_seqlens_k.numel() >= 2,
                "cu_seqlens_q/cu_seqlens_k must have at least 2 elements");
    const int64_t batch_size = cu_seqlens_q.numel() - 1;
    TORCH_CHECK(cu_seqlens_k.numel() == batch_size || cu_seqlens_k.numel() == batch_size + 1,
                "cu_seqlens_k must have shape [batch_size] (lengths) or [batch_size + 1] (cumulative)");
    TORCH_CHECK(k.size(0) == v.size(0), "k and v total tokens must match");
    TORCH_CHECK(k.size(1) == v.size(1), "k and v num_heads must match");
    TORCH_CHECK(k.size(2) == v.size(2), "k and v head_dim must match");
    TORCH_CHECK(q.size(2) == k.size(2), "q/k/v head_dim must match");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");

    const int64_t num_heads = q.size(1);
    const int64_t num_heads_k = k.size(1);
    const int64_t head_size = q.size(2);
    const bool is_seqlens_k_cumulative = cu_seqlens_k.numel() == batch_size + 1;

    torch::Tensor out = torch::zeros_like(q);
    torch::Tensor l = torch::zeros({batch_size, num_heads, max_seqlen_q}, q.options().dtype(torch::kFloat32));

    Flash_fwd_params params;
    set_params_fprop(
        params,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        num_heads_k,
        head_size,
        q, k, v, out, l,
        cu_seqlens_q.data_ptr(),
        cu_seqlens_k.data_ptr(),
        is_causal
    );
    params.is_seqlens_k_cumulative = is_seqlens_k_cumulative;

    run_mha_fwd(params);

    return {out, l};
}

std::vector<torch::Tensor>
mha_varlen_bwd(torch::Tensor q,
               torch::Tensor k,
               torch::Tensor v,
               torch::Tensor out,
               torch::Tensor l,
               torch::Tensor dout,
               torch::Tensor cu_seqlens_q,
               torch::Tensor cu_seqlens_k,
               int64_t max_seqlen_q,
               int64_t max_seqlen_k,
               bool is_causal)
{
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q, k, v must be rank-3 packed tensors");
    TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_k.is_cuda(), "cu_seqlens_q/cu_seqlens_k must be CUDA tensors");
    TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::kInt32 && cu_seqlens_k.scalar_type() == torch::kInt32,
                "cu_seqlens_q/cu_seqlens_k must be int32 tensors");
    TORCH_CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous(),
                "cu_seqlens_q/cu_seqlens_k must be contiguous");
    TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "cu_seqlens_q/cu_seqlens_k must be rank-1");
    TORCH_CHECK(cu_seqlens_q.numel() >= 2 && cu_seqlens_k.numel() >= 2,
                "cu_seqlens_q/cu_seqlens_k must have at least 2 elements");
    const int64_t batch_size = cu_seqlens_q.numel() - 1;
    TORCH_CHECK(cu_seqlens_k.numel() == batch_size || cu_seqlens_k.numel() == batch_size + 1,
                "cu_seqlens_k must have shape [batch_size] (lengths) or [batch_size + 1] (cumulative)");
    TORCH_CHECK(k.size(0) == v.size(0), "k and v total tokens must match");
    TORCH_CHECK(k.size(1) == v.size(1), "k and v num_heads must match");
    TORCH_CHECK(k.size(2) == v.size(2), "k and v head_dim must match");
    TORCH_CHECK(q.size(2) == k.size(2), "q/k/v head_dim must match");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "num_heads_q must be divisible by num_heads_k for GQA/MQA");

    TORCH_CHECK(out.sizes() == q.sizes(), "out must match q shape");
    TORCH_CHECK(dout.sizes() == q.sizes(), "dout must match q shape");
    TORCH_CHECK(l.dim() == 3, "l must be rank-3 for varlen_bwd");

    const int64_t num_heads = q.size(1);
    const int64_t num_heads_k = k.size(1);
    const int64_t head_size = q.size(2);
    const int64_t total_k = k.size(0);
    const bool is_seqlens_k_cumulative = cu_seqlens_k.numel() == batch_size + 1;
    TORCH_CHECK(l.size(0) == batch_size && l.size(1) == num_heads && l.size(2) == max_seqlen_q,
                "l must have shape [batch_size, nheads_q, max_seqlen_q]");

    torch::Tensor dq = torch::zeros_like(q);
    torch::Tensor dk = torch::zeros_like(k);
    torch::Tensor dv = torch::zeros_like(v);
    torch::Tensor dk_expanded = dk;
    torch::Tensor dv_expanded = dv;
    if (num_heads != num_heads_k) {
        dk_expanded = torch::zeros({total_k, num_heads, head_size}, k.options().dtype(torch::kFloat16));
        dv_expanded = torch::zeros({total_k, num_heads, head_size}, v.options().dtype(torch::kFloat16));
    }
    torch::Tensor do_o = torch::zeros(l.sizes(), l.options());

    Flash_bwd_params params;
    set_params_dgrad(
        params,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        num_heads_k,
        head_size,
        q, k, v, out, l, dout,
        dq, dk_expanded, dv_expanded, do_o,
        cu_seqlens_q.data_ptr(),
        cu_seqlens_k.data_ptr(),
        is_causal
    );
    params.is_seqlens_k_cumulative = is_seqlens_k_cumulative;

    run_mha_bwd(params);

    if (num_heads != num_heads_k) {
        torch::sum_out(
            dk,
            torch::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}),
            {2}
        );
        torch::sum_out(
            dv,
            torch::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}),
            {2}
        );
    }

    return {dq, dk, dv};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("varlen_fwd", &mha_varlen_fwd, "Varlen forward pass");
    m.def("varlen_bwd", &mha_varlen_bwd, "Varlen backward pass");
}
