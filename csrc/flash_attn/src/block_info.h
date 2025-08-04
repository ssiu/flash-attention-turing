#pragma once

struct BlockInfo {

    __device__ BlockInfo(int seq_len, const int bidb)
        : sum_s(-1)
        , actual_seq_len(seq_len)
    {}

    __forceinline__ __device__ int q_offset(const int batch_stride, const int bidb) const {
        return bidb * batch_stride;
    }

    const int sum_s;
    const int actual_seq_len;
};



