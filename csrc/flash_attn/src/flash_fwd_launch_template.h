
#pragma once
#include "flash.h"
#include "flash_fwd_kernel.cu"
#include <cutlass/numeric_types.h>
using half_t = cutlass::half_t;


template <typename Kernel_traits, bool Is_causal, typename Params>
__global__ __launch_bounds__(256)
void flash_fwd_kernel(Params params) {

}


template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params params) {

    int batch_size = params.b;
    int num_heads = params.h;
    int seq_len = params.seqlen_q;

    //auto kernel = flash_fwd_kernel<Kernel_traits, Is_causal>;

    constexpr int kBlockM = Kernel_traits::kBlockM;

    dim3 dimGrid(batch_size, num_heads, seq_len / kBlockM);
    dim3 dimBlock(256);
    int maxbytes = 65536;


    cudaFuncSetAttribute(flash_fwd_kernel<Kernel_traits, Is_causal, Flash_fwd_params>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);


    flash_fwd_kernel<Kernel_traits, Is_causal, Flash_fwd_params><<<dimGrid, dimBlock, maxbytes>>>(params);

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
