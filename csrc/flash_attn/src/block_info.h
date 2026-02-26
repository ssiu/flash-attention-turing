#pragma once

struct BlockInfo {

    __forceinline__ __device__ static int prefix_sum_lengths(const int *lengths, int end) {
        int sum = 0;
        for (int i = 0; i < end; ++i) { sum += lengths[i]; }
        return sum;
    }

    __device__ BlockInfo(int seqlen_q, int seqlen_k, const int bidb,
                         const int *cu_seqlens_q = nullptr,
                         const int *cu_seqlens_k = nullptr,
                         bool is_seqlens_k_cumulative = true)
        : sum_s_q(cu_seqlens_q == nullptr ? -1 : cu_seqlens_q[bidb])
        , sum_s_k(cu_seqlens_k == nullptr ? -1 : (is_seqlens_k_cumulative ? cu_seqlens_k[bidb] : prefix_sum_lengths(cu_seqlens_k, bidb)))
        , actual_seqlen_q(cu_seqlens_q == nullptr ? seqlen_q : (cu_seqlens_q[bidb + 1] - cu_seqlens_q[bidb]))
        , actual_seqlen_k(cu_seqlens_k == nullptr ? seqlen_k : (is_seqlens_k_cumulative ? (cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb]) : cu_seqlens_k[bidb]))
    {}

    __forceinline__ __device__ int q_offset(const int row_stride, const int bidb) const {
        return sum_s_q == -1 ? bidb * actual_seqlen_q * row_stride : sum_s_q * row_stride;
    }

    __forceinline__ __device__ int k_offset(const int row_stride, const int bidb) const {
        return sum_s_k == -1 ? bidb * actual_seqlen_k * row_stride : sum_s_k * row_stride;
    }

    const int sum_s_q;
    const int sum_s_k;
    const int actual_seqlen_q;
    const int actual_seqlen_k;
};
