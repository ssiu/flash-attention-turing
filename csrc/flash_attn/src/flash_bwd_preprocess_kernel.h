#pragma once

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <float.h>
#include <torch/extension.h>
#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "kernel_traits.h"
#include "utils.h"


using namespace cute;

template <typename Kernel_traits, bool Is_causal>
inline __device__ void compute_dot_do_o(half_t* o_ptr,
                      half_t* do_ptr,
                      float*  d_ptr,
                      int batch_size, int seq_len, int num_heads, int head_dim, int is_causal)
{
    // o_offset: (batch_size, seq_len, num_heads, head_dim)
    // do_offset: (batch_size, seq_len, num_heads, head_dim)
    // d_offset: (batch_size, num_heads, seq_len)


    // block x = batch_size
    // block y = num_heads
    // block z = seq_len / 32

    // each thread loads 4 elements from do and o

    half_t rdO[4];
    half_t rO[4];
    float sum = 0;

    //int thread_id = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int do_o_offset = blockIdx.x * seq_len * num_heads * head_dim + blockIdx.z * 32 * num_heads * head_dim + blockIdx.y * head_dim;
    int d_offset = blockIdx.x * num_heads * seq_len + blockIdx.y * seq_len + blockIdx.z * 32;

    // each threadblock computes a 32 x headdim block
    // each warp computes a single row
    // for headdim = 64, each thread computes 2 elements
    // for headdim = 128, each thread computes 4 elements
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    int elements_per_thread = 0;
    if constexpr (kHeadDim == 64) {
        elements_per_thread = 2;
    } else {
        elements_per_thread = 4;
    }


    int thread_row = warp_id;
    int thread_col = lane_id * elements_per_thread;

    for (int i=0;i<elements_per_thread;i++) {
        rO[i] = o_ptr[do_o_offset + thread_row * num_heads * head_dim + thread_col + i];
        rdO[i] = do_ptr[do_o_offset + thread_row * num_heads * head_dim + thread_col + i];
    }

    // thread reduction
    for (int i=0;i<elements_per_thread;i ++) {
        sum += static_cast<float>(rO[i]) * static_cast<float>(rdO[i]);

    }


    // warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
       d_ptr[d_offset + thread_row] = sum;
    }

}