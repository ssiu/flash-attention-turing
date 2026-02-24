#pragma once

struct BlockInfo {

    __device__ BlockInfo(int seqlen_q, int seqlen_k, const int bidb)
        : sum_s_q(-1)
        , sum_s_k(-1)
        , actual_seqlen_q(seqlen_q)
        , actual_seqlen_k(seqlen_k)
    {}

    __forceinline__ __device__ int q_offset(const int batch_stride, const int bidb) const {
        return bidb * batch_stride;
    }

    __forceinline__ __device__ int k_offset(const int batch_stride, const int bidb) const {
        return bidb * batch_stride;
    }

    const int sum_s_q;
    const int sum_s_k;
    const int actual_seqlen_q;
    const int actual_seqlen_k;
};
