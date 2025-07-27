#pragma once

#include <cutlass/numeric_types.h>
using half_t = cutlass::half_t;

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    half_t *__restrict__ q_ptr;
    half_t *__restrict__ k_ptr;
    half_t *__restrict__ v_ptr;

    // The number of heads.
    int h;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    half_t * __restrict__ o_ptr;

    // The pointer to the softmax sum.
    float * __restrict__ l_ptr;
    int b, seqlen, d;
    bool is_causal;
};


struct Flash_bwd_params : public Flash_fwd_params {

    half_t *__restrict__ do_ptr;
    half_t *__restrict__ dq_ptr;
    half_t *__restrict__ dk_ptr;
    half_t *__restrict__ dv_ptr;
    float *__restrict__ do_o_ptr;


};


template<int Headdim, bool Is_causal> void run_mha_fwd_(Flash_fwd_params &params);


template<int Headdim, bool Is_causal> void run_mha_bwd_(Flash_bwd_params &params);