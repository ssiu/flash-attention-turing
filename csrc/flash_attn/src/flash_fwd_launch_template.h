
#pragma once
#include "flash.h"
#include "flash_fwd_kernel.cu"
#include <cutlass/numeric_types.h>
using half_t = cutlass::half_t;



template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd(half_t* q,
                   half_t* k,
                   half_t* v,
                   half_t* o,
                   float* l,
                   int batch_size,
                   int seq_len,
                   int num_heads,
                   int head_dim,
                   int is_causal) {


    //auto kernel = flash_fwd_kernel<Kernel_traits, Is_causal>;

    constexpr int kBlockM = Kernel_traits::kBlockM;

    dim3 dimGrid(batch_size, num_heads, seq_len / kBlockM);
    dim3 dimBlock(256);
    int maxbytes = 65536;


    cudaFuncSetAttribute(flash_fwd_kernel<Kernel_traits, Is_causal>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);


    flash_fwd_kernel<Kernel_traits, Is_causal><<<dimGrid, dimBlock, maxbytes>>>(q,
                                            k,
                                            v,
                                            o,
                                            l,
                                            batch_size, seq_len, num_heads, head_dim, is_causal);

}



template<bool Is_causal>
void run_mha_fwd_hdim128(half_t* q,
                        half_t* k,
                        half_t* v,
                        half_t* o,
                        float* l,
                        int batch_size,
                        int seq_len,
                        int num_heads,
                        int head_dim,
                        int is_causal) {
    constexpr static int Headdim = 128;
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 64;
    constexpr static int kNWarps = 8;
    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(q, k, v, o, l,
                                                                            batch_size, seq_len, num_heads, head_dim, is_causal);


}


