#pragma once

#include <cutlass/numeric_types.h>
using half_t = cutlass::half_t;


struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,
};


struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;


    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, d, seqlen_q_rounded, seqlen_k_rounded;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;

    bool is_causal;

};



template<int Headdim, bool Is_causal> void run_mha_fwd_(Flash_fwd_params &params);


template<int Headdim, bool Is_causal> void run_mha_bwd_(half_t* q,
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
                                                        int batch_size,
                                                        int seq_len,
                                                        int num_heads,
                                                        int head_dim,
                                                        int is_causal);