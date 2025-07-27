
#pragma once
#include "flash.h"
#include "flash_bwd_kernel.cu"
#include <cutlass/numeric_types.h>
using half_t = cutlass::half_t;



template<typename Kernel_traits, bool Is_causal>
void run_flash_bwd(Flash_bwd_params &params) {


    //auto kernel = flash_bwd_kernel<Kernel_traits, Is_causal>;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;

    // compute dO \circ O
    // we use 1 warp to compute a single row
    // each thread block we launch 1024 = 32 x 32 threads = 32 warps
    // so each thread block process 32 rows
    dim3 dimGrid_dot_do_o(params.b, params.h, params.seqlen / 32);
    dim3 dimBlock_dot_do_o(1024);
    compute_dot_do_o<Kernel_traits, Is_causal><<<dimGrid_dot_do_o, dimBlock_dot_do_o>>>(params.o_ptr,
                    params.do_ptr,
                    params.do_o_ptr,
                    params.b, params.seqlen, params.h, params.d, is_causal);


    int maxbytes = 65536;

    // compute dQ
    dim3 dimGrid_dq(params.b, params.h, params.seqlen / kBlockM);
    dim3 dimBlock_dq(256);

    //auto dq_kernel = compute_dq_kernel<Kernel_traits, Is_causal>;
    cudaFuncSetAttribute(compute_dq_kernel<Kernel_traits, Is_causal>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    compute_dq_kernel<Kernel_traits, Is_causal><<<dimGrid_dq, dimBlock_dq, maxbytes>>>(params.q_ptr,
                                            params.k_ptr,
                                            params.v_ptr,
                                            params.l_ptr,
                                            params.do_o_ptr,
                                            params.do_ptr,
                                            params.dq_ptr,
                                            params.b, params.seqlen, params.h, params.d, is_causal);


    // compute dK, dV
    dim3 dimGrid_dk_dv(params.b, params.h, params.seqlen / kBlockN);
    dim3 dimBlock_dk_dv(256);

    //auto dk_dv_kernel = compute_dk_dv_kernel<Kernel_traits, Is_causal>;
    cudaFuncSetAttribute(compute_dk_dv_kernel<Kernel_traits, Is_causal>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    compute_dk_dv_kernel<Kernel_traits, Is_causal><<<dimGrid_dk_dv, dimBlock_dk_dv, maxbytes>>>(params.q_ptr,
                                            params.k_ptr,
                                            params.v_ptr,
                                            params.l_ptr,
                                            params.do_o_ptr,
                                            params.do_ptr,
                                            params.dk_ptr,
                                            params.dv_ptr,
                                            params.b, params.seqlen, params.h, params.d, is_causal);

//    // compute dQ, dK, dV in a single kernel
//    dim3 dimGrid_dq_dk_dv(params.b, params.h, params.seqlen / kBlockN);
//    dim3 dimBlock_dq_dk_dv(256);
//
//    //auto dk_dv_kernel = compute_dk_dv_kernel<Kernel_traits, Is_causal>;
//    cudaFuncSetAttribute(compute_dq_dk_dv_kernel<Kernel_traits, Is_causal>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
//    compute_dq_dk_dv_kernel<Kernel_traits, Is_causal><<<dimGrid_dq_dk_dv, dimBlock_dq_dk_dv, maxbytes>>>(q,
//                                            k,
//                                            v,
//                                            l,
//                                            d,
//                                            do_,
//                                            dq_float,
//                                            dq,
//                                            dk,
//                                            dv,
//                                            params.b, params.seqlen, params.h, params.d, is_causal);
//
//    // convert dq from float to half
//    dim3 dimGrid_convert_dq(params.b, params.h, params.seqlen / kBlockM);
//    dim3 dimBlock_convert_dq(256);
//    convert_dq<Kernel_traits, Is_causal><<<dimGrid_convert_dq, dimBlock_convert_dq>>>(
//                    dq_float,
//                    dq,
//                    params.b, params.seqlen, params.h, params.d, is_causal);




}


template<bool Is_causal>
void run_mha_bwd_hdim64(Flash_bwd_params &params) {
    constexpr static int Headdim = 64;
    constexpr static int kBlockM = 64;
    constexpr static int kBlockN = 64;
    constexpr static int kNWarps = 8;

    run_flash_bwd<Flash_bwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);


}



template<bool Is_causal>
void run_mha_bwd_hdim128(Flash_bwd_params &params) {
    constexpr static int Headdim = 128;
    constexpr static int kBlockM = 64;
    constexpr static int kBlockN = 64;
    constexpr static int kNWarps = 8;

    run_flash_bwd<Flash_bwd_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps>, Is_causal>(params);


}


