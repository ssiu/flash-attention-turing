
#pragma once
#include "flash.h"
#include "flash_bwd_kernel.cu"
#include <cutlass/numeric_types.h>
using half_t = cutlass::half_t;



template<typename Kernel_traits, bool Is_causal>
void run_flash_bwd(half_t* q,
                   half_t* k,
                   half_t* v,
                   half_t* o,
                    float* l,
                    float* d,
                    half_t* do_,
                    half_t* dq,
                    half_t* dk,
                    half_t* dv,
                   int batch_size,
                   int seq_len,
                   int num_heads,
                   int head_dim,
                   int is_causal) {


    //auto kernel = flash_bwd_kernel<Kernel_traits, Is_causal>;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;

    // compute dO \circ O
    // we use 1 warp to compute a single row
    // each thread block we launch 1024 = 32 x 32 threads = 32 warps
    // so each thread block process 32 rows
    dim3 dimGrid_dot_do_o(batch_size, num_heads, seq_len / 32);
    dim3 dimBlock_dot_do_o(1024);
    compute_dot_do_o<Kernel_traits, Is_causal><<<dimGrid_dot_do_o, dimBlock_dot_do_o>>>(o,
                    do_,
                    d,
                    batch_size, seq_len, num_heads, head_dim, is_causal);


    int maxbytes = 65536;

    // compute dQ
    dim3 dimGrid_dq(batch_size, num_heads, seq_len / kBlockM);
    dim3 dimBlock_dq(256);

    //auto dq_kernel = compute_dq_kernel<Kernel_traits, Is_causal>;
    cudaFuncSetAttribute(compute_dq_kernel<Kernel_traits, Is_causal>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    compute_dq_kernel<Kernel_traits, Is_causal><<<dimGrid_dq, dimBlock_dq, maxbytes>>>(q,
                                            k,
                                            v,
                                            l,
                                            d,
                                            do_,
                                            dq,
                                            batch_size, seq_len, num_heads, head_dim, is_causal);


    // compute dK, dV
    dim3 dimGrid_dk_dv(batch_size, num_heads, seq_len / kBlockN);
    dim3 dimBlock_dk_dv(256);

    //auto dk_dv_kernel = compute_dk_dv_kernel<Kernel_traits, Is_causal>;
    cudaFuncSetAttribute(compute_dk_dv_kernel<Kernel_traits, Is_causal>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    compute_dk_dv_kernel<Kernel_traits, Is_causal><<<dimGrid_dk_dv, dimBlock_dk_dv, maxbytes>>>(q,
                                            k,
                                            v,
                                            l,
                                            d,
                                            do_,
                                            dk,
                                            dv,
                                            batch_size, seq_len, num_heads, head_dim, is_causal);

}



template<bool Is_causal>
void run_mha_bwd_hdim128(half_t* q,
                        half_t* k,
                        half_t* v,
                        half_t* o,
                        float* l,
                        float* d,
                        half_t* do_,
                        half_t* dq,
                        half_t* dk,
                        half_t* dv,
                        int batch_size,
                        int seq_len,
                        int num_heads,
                        int head_dim,
                        int is_causal) {
    constexpr static int Headdim = 128;
    constexpr static int kBlockM = 64;
    constexpr static int kBlockN = 64;
    constexpr static int kNWarps = 8;

    run_flash_bwd<Flash_bwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(q, k, v, o, l, d, do_, dq, dk, dv,
                                                                            batch_size, seq_len, num_heads, head_dim, is_causal);


}


