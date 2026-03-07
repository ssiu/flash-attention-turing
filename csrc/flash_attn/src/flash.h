#pragma once

#include <cutlass/numeric_types.h>
using half_t = cutlass::half_t;

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    half_t *__restrict__ q_ptr;
    half_t *__restrict__ k_ptr;
    half_t *__restrict__ v_ptr;

    // index_t q_batch_stride;
    // index_t k_batch_stride;
    // index_t v_batch_stride;
    // index_t q_row_stride;
    // index_t k_row_stride;
    // index_t v_row_stride;
    // index_t q_head_stride;
    // index_t k_head_stride;
    // index_t v_head_stride;

    // The number of heads.
    int h;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_k;
    int h_h_k_ratio;

    // Optional varlen cumulative sequence lengths (device pointers, int32).
    int *__restrict__ cu_seqlens_q;
    int *__restrict__ cu_seqlens_k;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    half_t * __restrict__ o_ptr;


    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;


    // The pointer to the softmax sum.
    float * __restrict__ l_ptr;
    int b, seqlen_q, seqlen_k, d;
    bool is_causal;
};


struct Flash_bwd_params : public Flash_fwd_params {

    half_t *__restrict__ do_ptr;
    half_t *__restrict__ dq_ptr;
    half_t *__restrict__ dk_ptr;
    half_t *__restrict__ dv_ptr;
    float *__restrict__ do_o_ptr;

    // index_t do_batch_stride;
    // index_t do_row_stride;
    // index_t do_head_stride;
    // index_t dq_batch_stride;
    // index_t dk_batch_stride;
    // index_t dv_batch_stride;
    // index_t dq_row_stride;
    // index_t dk_row_stride;
    // index_t dv_row_stride;
    // index_t dq_head_stride;
    // index_t dk_head_stride;
    // index_t dv_head_stride;

};


template<int Headdim, bool Is_causal> void run_mha_fwd_(Flash_fwd_params &params);


template<int Headdim, bool Is_causal> void run_mha_bwd_(Flash_bwd_params &params);
