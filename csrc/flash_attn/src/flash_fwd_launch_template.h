
#pragma once
#include "flash.h"
#include "flash_fwd_kernel.h"
#include <cutlass/numeric_types.h>
#include "static_switch.h"
using half_t = cutlass::half_t;

template <typename Kernel_traits, bool Is_causal, bool Is_even_MN>
__global__ __launch_bounds__(256)
// for some reason changing this into params struc is 10% slower for hdim = 128
void flash_fwd_kernel(half_t* __restrict__ q,
                          half_t* __restrict__ k,
                          half_t* __restrict__ v,
                          half_t* __restrict__ o,
                          float* __restrict__ l,
                          int batch_size,
                          int seqlen_q,
                          int seqlen_k,
                          int num_heads,
                          int num_heads_k,
                          int h_h_k_ratio,
                          int head_dim,
                          int is_casual)
{
    compute_attn<Kernel_traits, Is_causal, Is_even_MN>(q,
                                           k,
                                           v,
                                           o,
                                           l,
                                           batch_size,
                                           seqlen_q,
                                           seqlen_k,
                                           num_heads,
                                           num_heads_k,
                                           h_h_k_ratio,
                                           head_dim,
                                           is_casual);
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params) {


    //auto kernel = flash_fwd_kernel<Kernel_traits, Is_causal>;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;

    const int num_m_block = (params.seqlen_q + kBlockM - 1) / kBlockM;

    const bool is_even_MN  = params.seqlen_q % kBlockM == 0 && params.seqlen_k % kBlockN == 0;

    dim3 dimGrid(num_m_block, params.b, params.h);
    dim3 dimBlock(256);
    int maxbytes = 65536;


    BOOL_SWITCH(is_even_MN, Is_even_MN, [&] {
        cudaFuncSetAttribute(flash_fwd_kernel<Kernel_traits, Is_causal, Is_even_MN>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
        flash_fwd_kernel<Kernel_traits, Is_causal, Is_even_MN><<<dimGrid, dimBlock, maxbytes>>>(params.q_ptr,
                                                                                    params.k_ptr,
                                                                                    params.v_ptr,
                                                                                    params.o_ptr,
                                                                                    params.l_ptr,
                                                                                    params.b,
                                                                                    params.seqlen_q,
                                                                                    params.seqlen_k,
                                                                                    params.h,
                                                                                    params.h_k,
                                                                                    params.h_h_k_ratio,
                                                                                    params.d,
                                                                                    params.is_causal);

    });

}



template<bool Is_causal>
void run_mha_fwd_hdim128(Flash_fwd_params &params) {
    constexpr static int Headdim = 128;
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 64;
    constexpr static int kNWarps = 8;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);


}


template<bool Is_causal>
void run_mha_fwd_hdim64(Flash_fwd_params &params) {
    constexpr static int Headdim = 64;
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 128;
    constexpr static int kNWarps = 8;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);


}
