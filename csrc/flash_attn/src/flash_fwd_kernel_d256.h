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

#include "block_info.h"
#include "kernel_traits_d256.h"
// Shared legacy helpers expect CuTe names in their including scope.
using namespace cute;
#include "utils.h"
#include "mask.h"

namespace flash_attn_d256 {

using namespace cute;


// for some reason changing this into params struct is 15% slower for hdim = 128
// Different complete warp roles deliberately use non-aligned named barriers.
template<int Id,int Count>
__device__ __forceinline__ void role_barrier() {
    asm volatile("barrier.sync %0, %1;" :: "n"(Id), "n"(Count) : "memory");
}

// Probabilities use two aligned four-word vectors per producer/consumer.
__device__ __forceinline__ void store_probability4(uint32_t* ptr,
        uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(address), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}
__device__ __forceinline__ void load_probability4(const uint32_t* ptr,
        uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(address) : "memory");
}

template <typename Kernel_traits, bool Is_causal, bool Is_even_MN>
inline __device__ void compute_attn_1rowblock(
                          const half_t* __restrict__ q,
                          const half_t* __restrict__ k,
                          const half_t* __restrict__ v,
                          half_t* __restrict__ o,
                          float* __restrict__ l,
                          const int* __restrict__ cu_seqlens_q,
                          const int* __restrict__ cu_seqlens_k,
                          const int batch_size,
                          const int max_seqlen_q,
                          const int max_seqlen_k,
                          const int num_heads,
                          const int num_heads_k,
                          const int h_h_k_ratio,
                          const int head_dim,
                          const float softmax_scale,
                          const int is_casual,
                          const int bidb,
                          const int bidh,
                          const int m_block)
{
    static_assert(Kernel_traits::kHeadDim==256 && Kernel_traits::kBlockM==64 &&
                  Kernel_traits::kBlockN==32 && Kernel_traits::kNWarps==4,
                  "D256 requires256 threads: four QK/softmax and four PV warps");
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;


    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(max_seqlen_q, max_seqlen_k, bidb, cu_seqlens_q, cu_seqlens_k);
    const int seqlen_q = binfo.actual_seqlen_q;
    const int seqlen_k = binfo.actual_seqlen_k;


    if (m_block * kBlockM >= seqlen_q) {
        return;
    }


    Tensor mQ = make_tensor(make_gmem_ptr(q + binfo.q_offset(num_heads * head_dim, bidb)),
                            make_shape(seqlen_q, num_heads, head_dim),
                            make_stride(num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));




    Tensor mK = make_tensor(make_gmem_ptr(k + binfo.k_offset(num_heads_k * head_dim, bidb)),
                            make_shape(seqlen_k, num_heads_k, head_dim),
                            make_stride(num_heads_k * head_dim, head_dim, Int<1>{}));

    Tensor gK = local_tile(mK(_, bidh / h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));



    Tensor mV = make_tensor(make_gmem_ptr(v + binfo.k_offset(num_heads_k * head_dim, bidb)),
                            make_shape(seqlen_k, num_heads_k, head_dim),
                            make_stride(num_heads_k * head_dim, head_dim, Int<1>{}));

    Tensor gV = local_tile(mV(_, bidh / h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));


    Tensor mO = make_tensor(make_gmem_ptr(o + binfo.q_offset(num_heads * head_dim, bidb)),
                            make_shape(seqlen_q, num_heads, head_dim),
                            make_stride(num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));

    // L = m + log l
    Tensor mL = make_tensor(make_gmem_ptr(reinterpret_cast<float*>(l)),
                             make_shape(batch_size, num_heads, max_seqlen_q),
                             make_stride(max_seqlen_q * num_heads, max_seqlen_q, Int<1>{}));

    Tensor gL = local_tile(mL(bidb, bidh, _), Shape<Int<kBlockM>>{},
                           make_coord(m_block));
    const int n_block_min = 0;
    int m_block_max = ceil_div(seqlen_q, kBlockM);
    int n_block_max = ceil_div(seqlen_k, kBlockN);


    int n_masking_steps = (!Is_causal)
        ? 1
        : ((Is_even_MN && Is_causal) ? ceil_div(kBlockM, kBlockN) : ceil_div(kBlockM, kBlockN) + 1);

    int causal_offset = 0;
    int is_even_mn_offset = 0;

    if constexpr(Is_causal) {

        n_block_max = fmaxf(0, ceil_div((m_block + 1) * kBlockM + seqlen_k - seqlen_q, kBlockN));
        n_masking_steps = fminf(n_masking_steps, n_block_max);

        causal_offset = seqlen_k - seqlen_q - (n_block_max - 1) * kBlockN + m_block * kBlockM;

    }

    is_even_mn_offset = seqlen_k - (n_block_max - 1) * kBlockN;

    // if seqlen_q > seqlen_k we exit early for the blocks with rows that are fully masked
    if (n_block_max == 0) {return;}

    extern __shared__ __align__(16) char smem_[];
    auto smem_half = reinterpret_cast<half_t*>(smem_);
    auto sP = reinterpret_cast<uint32_t*>(smem_ + 49152);
    auto sAlpha = reinterpret_cast<float*>(smem_ + 57344);
    auto sFinalL = reinterpret_cast<float*>(smem_ + 57856);
    typename Kernel_traits::TiledMma tiled_mma;

    if (threadIdx.x < 128) {
        const int logical_tid = threadIdx.x;
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int thread_row = warp_id * 16 + lane_id / 4;
    const int global_row_offset = m_block * kBlockM;

    float rM_old[2] = {-FLT_MAX, -FLT_MAX};
    float rM[2] = {0.0f};
    float rL_old[2] = {0.0f};
    float rL[2] = {0.0f};
    // for storing rowsum(P)
    float rD[2] = {0.0f};

    unsigned mask;
    if (lane_id < 4)       mask = 0x0000000F;  // Lanes  0 -  3
    else if (lane_id < 8)  mask = 0x000000F0;  // Lanes  4 -  7
    else if (lane_id < 12) mask = 0x00000F00;  // Lanes  8 - 11
    else if (lane_id < 16) mask = 0x0000F000;  // Lanes 12 - 15
    else if (lane_id < 20) mask = 0x000F0000;  // Lanes 16 - 19
    else if (lane_id < 24) mask = 0x00F00000;  // Lanes 20 - 23
    else if (lane_id < 28) mask = 0x0F000000;  // Lanes 24 - 27
    else                   mask = 0xF0000000;  // Lanes 28 - 31


    int lane_id_to_read_from;
    if (lane_id < 4)       lane_id_to_read_from = 0;   // Lanes  0 -  3
    else if (lane_id < 8)  lane_id_to_read_from = 4;   // Lanes  4 -  7
    else if (lane_id < 12) lane_id_to_read_from = 8;   // Lanes  8 - 11
    else if (lane_id < 16) lane_id_to_read_from = 12;  // Lanes 12 - 15
    else if (lane_id < 20) lane_id_to_read_from = 16;  // Lanes 16 - 19
    else if (lane_id < 24) lane_id_to_read_from = 20;  // Lanes 20 - 23
    else if (lane_id < 28) lane_id_to_read_from = 24;  // Lanes 24 - 27
    else                   lane_id_to_read_from = 28;  // Lanes 28 - 31

        Tensor sQ = make_tensor(make_smem_ptr(smem_half), typename Kernel_traits::SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(smem_half), typename Kernel_traits::SmemLayoutK{});
        typename Kernel_traits::GmemTiledCopyQK gmem_tiled_copy_QK;
        typename Kernel_traits::GmemTiledCopyV gmem_tiled_copy_V;
        auto thr_copy_QK = gmem_tiled_copy_QK.get_slice(logical_tid);
        auto thr_copy_V = gmem_tiled_copy_V.get_slice(logical_tid);
        auto tQgQ = thr_copy_QK.partition_S(gQ);
        auto tQsQ = thr_copy_QK.partition_D(sQ);
        auto tKgK = thr_copy_QK.partition_S(gK);
        auto tKsK = thr_copy_QK.partition_D(sK);
        auto tVgV = thr_copy_V.partition_S(gV);
        auto cQ_identity = make_identity_tensor(make_shape(Int<kBlockM>{},Int<kHeadDim>{}));
        auto cK_identity = make_identity_tensor(make_shape(Int<kBlockN>{},Int<kHeadDim>{}));
        auto tCqQ = thr_copy_QK.partition_S(cQ_identity);
        auto tCqK = thr_copy_QK.partition_S(cK_identity);
        auto thr_mma_S = tiled_mma.get_slice(logical_tid);
        auto tSsQ = thr_mma_S.partition_A(sQ);
        auto tSsK = thr_mma_S.partition_B(sK);
        auto tSrQ = thr_mma_S.make_fragment_A(tSsQ);
        auto tSrK = thr_mma_S.make_fragment_B(tSsK);
        auto tSrS_float = partition_fragment_C(tiled_mma,Shape<Int<kBlockM>,Int<kBlockN>>{});
        auto s2r_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQ{},tiled_mma);
        auto s2r_thr_copy_Q = s2r_tiled_copy_Q.get_slice(logical_tid);
        auto tSsQ_copy_view = s2r_thr_copy_Q.partition_S(sQ);
        auto tSrQ_copy_view = s2r_thr_copy_Q.retile_D(tSrQ);
        auto s2r_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomK{},tiled_mma);
        auto s2r_thr_copy_K = s2r_tiled_copy_K.get_slice(logical_tid);
        auto tSsK_copy_view = s2r_thr_copy_K.partition_S(sK);
        auto tSrK_copy_view = s2r_thr_copy_K.retile_D(tSrK);
        auto QK_BLOCK_MAX = size<2>(tSsK);
        // Q prologue aliases future K/V0. Retire every reader before reuse.
        masked_copy<Is_even_MN>(gmem_tiled_copy_QK,tQgQ,tQsQ,tCqQ,
                               seqlen_q-m_block*kBlockM,/*clear_D=*/true);
        role_barrier<1,128>();
        CUTE_UNROLL
        for (int qk_block=0; qk_block<QK_BLOCK_MAX; ++qk_block)
            copy(s2r_tiled_copy_Q,tSsQ_copy_view(_,_,qk_block),tSrQ_copy_view(_,_,qk_block));
        role_barrier<0,256>();  // all Q readers retired; both roles start pipeline
        int n_block = n_block_max-1;
        int generation = 0;
        Mask<Is_causal> accum_s_mask(seqlen_q,seqlen_k);
        for (int masking_step=0; masking_step<n_masking_steps; ++masking_step,--n_block,++generation) {
            auto sV = make_tensor(make_smem_ptr(smem_half+8192+(generation&1)*8192),
                                 typename Kernel_traits::SmemLayoutV{});
            auto tVsV = thr_copy_V.partition_D(sV);
            // Physical row bounds/zero fill remain original masked-copy policy.
            masked_copy<Is_even_MN>(gmem_tiled_copy_QK,tKgK(_,_,_,n_block),tKsK,tCqK,
                                   seqlen_k-n_block*kBlockN,/*clear_D=*/true);
            masked_copy<Is_even_MN>(gmem_tiled_copy_QK,tVgV(_,_,_,n_block),tVsV,tCqK,
                                   seqlen_k-n_block*kBlockN,/*clear_D=*/true);
            role_barrier<1,128>();  // producer-only K/V publication

        clear(tSrS_float);
        CUTE_UNROLL
        for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
            copy(s2r_tiled_copy_K, tSsK_copy_view(_,_,qk_block), tSrK_copy_view(_,_,qk_block));

            gemm(tiled_mma, tSrQ(_,_,qk_block), tSrK(_,_,qk_block), tSrS_float);

        }
        // for now we rescale before we apply causal mask
        for (int i=0;i< tSrS_float.size();i ++ ) {
            tSrS_float[i] *= softmax_scale;
        }


        accum_s_mask.template apply_mask_fwd<Is_causal, Is_even_MN>(
            tSrS_float,
            warp_id,
            lane_id,
            kBlockN,
            causal_offset,
            is_even_mn_offset
        );



        // compute m = rowmax(S)
        for (int i=0; i< 2; i++) {
            rM[i] = rM_old[i];
        }


        // intra-thread reduction

        for (int i=0; i< 2; i++) {
            for (int j=0; j < tSrS_float(make_coord(_,i),_,_).size(); j++) {
                rM[i] = fmaxf(rM[i], tSrS_float(make_coord(_,i),_,_)[j]);
            }
        }


        // intra-warp reduction
        for (int i=0; i<2; i++) {
            for (int offset = 2; offset > 0; offset /= 2) {
                rM[i] = fmaxf(rM[i], __shfl_down_sync(mask, rM[i], offset));
            }
        }


        // sync rM

        for (int i =0; i<2; i++) {
            rM[i] = __shfl_sync(mask, rM[i], lane_id_to_read_from);
        }



        // compute P = softmax(S)
        for (int i =0; i<2; i++) {
            for (int j=0; j < tSrS_float(make_coord(_,i),_,_).size(); j++) {
                if (rM[i] == -FLT_MAX) {
                    tSrS_float(make_coord(_,i),_,_)[j] = 0.0f;
                } else {
                    tSrS_float(make_coord(_,i),_,_)[j] = expf(tSrS_float(make_coord(_,i),_,_)[j] - rM[i]);
                }

            }
        }



        // rescale l and also reset rD to 0
        for (int i =0; i<2; i++) {
            rL[i] = expf(rM_old[i] - rM[i]) * rL_old[i];
            rD[i] = 0.0f;
        }
        // compute sum(sP)

        // thread reduction

        for (int i =0; i<2; i++) {
            for (int j=0; j < tSrS_float(make_coord(_,i),_,_).size(); j++) {
                rD[i] += tSrS_float(make_coord(_,i),_,_)[j];
            }
        }



        // warp reduction
        for (int i =0; i<2; i++) {
            for (int offset = 2; offset > 0; offset /= 2) {
                rD[i] +=  __shfl_down_sync(mask, rD[i], offset);
            }
        }



        for (int i =0; i<2; i++) {
            rL[i] += rD[i];
        }




        // sync rL
        for (int i =0; i<2; i++) {
            rL[i] = __shfl_sync(mask, rL[i], lane_id_to_read_from);
        }



        Tensor tOrP = convert_type<half_t>(tSrS_float);

        // Retain adjacent FP16 probability pairs and conversion.
        CUTE_UNROLL
        for (int vector=0; vector<2; ++vector) {
            store_probability4(sP+(generation&1)*1024+vector*512+logical_tid*4,
                uint32_t(tOrP(8*vector).raw()) | (uint32_t(tOrP(8*vector+1).raw()) << 16),
                uint32_t(tOrP(8*vector+2).raw()) | (uint32_t(tOrP(8*vector+3).raw()) << 16),
                uint32_t(tOrP(8*vector+4).raw()) | (uint32_t(tOrP(8*vector+5).raw()) << 16),
                uint32_t(tOrP(8*vector+6).raw()) | (uint32_t(tOrP(8*vector+7).raw()) << 16));
        }
        if (lane_id % 4 == 0) {
            CUTE_UNROLL
            for (int i=0; i<2; ++i) {
                sAlpha[(generation&1)*64+thread_row+8*i] = expf(rM_old[i]-rM[i]);
                if (n_block == 0) sFinalL[thread_row+8*i] = rL[i];
            }
        }
        role_barrier<2,256>();  // READY: P/alpha and this V generation visible
        // update m and l
        for (int i = 0; i< 2;i++) {
            rM_old[i] = rM[i];
            rL_old[i] = rL[i];
        }

        }
        // Main-only K carry: masked prefix remains unchanged.
        auto tCarryK = make_fragment_like(tKsK);
        if (n_block >= n_block_min) {
            copy(gmem_tiled_copy_QK,tKgK(_,_,_,n_block),tCarryK);
        }
        CUTE_NO_UNROLL
        for (; n_block>=n_block_min; --n_block,++generation) {
            // Odd-tile copy path
            if constexpr (!Is_even_MN) {
            auto sV = make_tensor(make_smem_ptr(smem_half+8192+(generation&1)*8192),
                                 typename Kernel_traits::SmemLayoutV{});
            auto tVsV = thr_copy_V.partition_D(sV);
            copy(gmem_tiled_copy_QK,tCarryK,tKsK);
            copy(gmem_tiled_copy_V,tVgV(_,_,_,n_block),tVsV);
            role_barrier<1,128>();
            } else {  // Even-tile copy path
            copy(gmem_tiled_copy_QK,tCarryK,tKsK);
            role_barrier<1,128>();
            }  // End copy paths

        clear(tSrS_float);
        CUTE_UNROLL
        for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
            copy(s2r_tiled_copy_K, tSsK_copy_view(_,_,qk_block), tSrK_copy_view(_,_,qk_block));

            gemm(tiled_mma, tSrQ(_,_,qk_block), tSrK(_,_,qk_block), tSrS_float);

        }
        // Current QK has retired; next K is only written at next main top.
        if (n_block > n_block_min) {
            copy(gmem_tiled_copy_QK,tKgK(_,_,_,n_block-1),tCarryK);
        }
        for (int i=0;i< tSrS_float.size();i ++ ) {
            tSrS_float[i] *= softmax_scale;
        }



        // compute m = rowmax(S)
        for (int i=0; i< 2; i++) {
            rM[i] = rM_old[i];
        }


        // intra-thread reduction

        for (int i=0; i< 2; i++) {
            for (int j=0; j < tSrS_float(make_coord(_,i),_,_).size(); j++) {
                rM[i] = fmaxf(rM[i], tSrS_float(make_coord(_,i),_,_)[j]);
            }
        }


        // intra-warp reduction
        for (int i=0; i<2; i++) {
            for (int offset = 2; offset > 0; offset /= 2) {
               rM[i] = fmaxf(rM[i], __shfl_xor_sync(mask, rM[i], offset));
            }
        }




        // compute P = softmax(S)
        CUTE_UNROLL
        for (int i =0; i<2; i++) {
            //float max_scaled = rM[i] * float(M_LOG2E);
            CUTE_UNROLL
            for (int j=0; j < tSrS_float(make_coord(_,i),_,_).size(); j++) {
                tSrS_float(make_coord(_,i),_,_)[j] = expf(tSrS_float(make_coord(_,i),_,_)[j] - rM[i]);
                // using FMA instructions inside exp is slower
                //tSrS_float(make_coord(_,i),_,_)[j] = exp2f(tSrS_float(make_coord(_,i),_,_)[j] * float(M_LOG2E) - max_scaled);
            }

            rL[i] = expf(rM_old[i] - rM[i]) * rL_old[i];
            // rL[i] = exp2f(rM_old[i] * float(M_LOG2E) - max_scaled) * rL_old[i];
            rD[i] = 0.0f;
        }





        // compute sum(sP)

        // thread reduction

        for (int i =0; i<2; i++) {
            for (int j=0; j < tSrS_float(make_coord(_,i),_,_).size(); j++) {
                rD[i] += tSrS_float(make_coord(_,i),_,_)[j];
            }
        }



        // warp reduction
        for (int i =0; i<2; i++) {
            for (int offset = 2; offset > 0; offset /= 2) {
               rD[i] +=  __shfl_xor_sync(mask, rD[i], offset);
            }
        }



        for (int i =0; i<2; i++) {
            rL[i] += rD[i];
        }





        Tensor tOrP = convert_type<half_t>(tSrS_float);

        // Retain adjacent FP16 probability pairs and conversion.
        CUTE_UNROLL
        for (int vector=0; vector<2; ++vector) {
            store_probability4(sP+(generation&1)*1024+vector*512+logical_tid*4,
                uint32_t(tOrP(8*vector).raw()) | (uint32_t(tOrP(8*vector+1).raw()) << 16),
                uint32_t(tOrP(8*vector+2).raw()) | (uint32_t(tOrP(8*vector+3).raw()) << 16),
                uint32_t(tOrP(8*vector+4).raw()) | (uint32_t(tOrP(8*vector+5).raw()) << 16),
                uint32_t(tOrP(8*vector+6).raw()) | (uint32_t(tOrP(8*vector+7).raw()) << 16));
        }
        if (lane_id % 4 == 0) {
            CUTE_UNROLL
            for (int i=0; i<2; ++i) {
                sAlpha[(generation&1)*64+thread_row+8*i] = expf(rM_old[i]-rM[i]);
                if (n_block == 0) sFinalL[thread_row+8*i] = rL[i];
            }
        }
        role_barrier<2,256>();  // READY: P/alpha and this V generation visible
        // update m and l
        for (int i = 0; i< 2;i++) {
            rM_old[i] = rM[i];
            rL_old[i] = rL[i];
        }

        }
        role_barrier<4,256>();  // terminal: final PV retired before O aliases shared
        // Original natural-log LSE/zero-L rule, one producer per row.
        if (lane_id % 4 == 0) {
            if (global_row_offset+thread_row<seqlen_q)
                gL[thread_row] = rL[0]==0.0f ? 0.0f : rM[0]+logf(rL[0]);
            if (global_row_offset+thread_row+8<seqlen_q)
                gL[thread_row+8] = rL[1]==0.0f ? 0.0f : rM[1]+logf(rL[1]);
        }
        return;
    }

    const int logical_tid = threadIdx.x-128;
    const int thread_row = (logical_tid/32)*16+(logical_tid%32)/4;
    auto thr_mma_O = tiled_mma.get_slice(logical_tid);
    auto tOrO_float = partition_fragment_C(tiled_mma,Shape<Int<kBlockM>,Int<kHeadDim>>{});
    auto p_proto = partition_fragment_C(tiled_mma,Shape<Int<kBlockM>,Int<kBlockN>>{});
    auto tOrP = make_fragment_like<half_t>(p_proto);
    static_assert(alignof(decltype(tOrP)) >= alignof(uint32_t));
    auto p_words = recast<uint32_t>(tOrP);
    static_assert(decltype(size(p_words))::value == 8);
    auto s2r_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomVt{},tiled_mma);
    auto s2r_thr_copy_V = s2r_tiled_copy_V.get_slice(logical_tid);
    clear(tOrO_float);
    role_barrier<0,256>();
    int generation = 0;
    // Odd-tile consumer path
    if constexpr (!Is_even_MN) {
    CUTE_NO_UNROLL
    for (int n_block=n_block_max-1; n_block>=n_block_min; --n_block,++generation) {
        role_barrier<2,256>();
        CUTE_UNROLL
        for (int vector=0; vector<2; ++vector) {
            load_probability4(sP+(generation&1)*1024+vector*512+logical_tid*4,
                p_words(4*vector), p_words(4*vector+1),
                p_words(4*vector+2), p_words(4*vector+3));
        }
        float row_alpha[2] = {sAlpha[(generation&1)*64+thread_row],sAlpha[(generation&1)*64+thread_row+8]};
        role_barrier<3,128>();  // consumers only; producer may advance after READY
        auto sVt = make_tensor(make_smem_ptr(smem_half+8192+(generation&1)*8192),
                              typename Kernel_traits::SmemLayoutVTransposed{});
        constexpr int PV_COLS = 32;
        CUTE_UNROLL
        for (int dc=0; dc<kHeadDim/PV_COLS; ++dc) {
            auto v_cols = local_tile(sVt, Shape<Int<PV_COLS>,Int<kBlockN>>{}, make_coord(dc,0));
            auto v_part = thr_mma_O.partition_B(v_cols);
            auto v_frag = thr_mma_O.make_fragment_B(v_part);
            auto v_src = s2r_thr_copy_V.partition_S(v_cols);
            auto v_dst = s2r_thr_copy_V.retile_D(v_frag);
            auto tile_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>,Int<PV_COLS>>{});
            // F01: same V load/fragment ownership; only high PV remains.
            CUTE_UNROLL
            for (int pb=0; pb<size<2>(v_part); ++pb) {
                copy(s2r_tiled_copy_V, v_src(_,_,pb), v_dst(_,_,pb));
            }
            clear(tile_o);
            CUTE_UNROLL
            for (int pb=0; pb<size<2>(v_part); ++pb) {
                gemm(tiled_mma, tOrP(_,_,pb), v_frag(_,_,pb), tile_o);
            }
            CUTE_UNROLL
            for (int n=0; n<size<2>(tile_o); ++n)
                CUTE_UNROLL
                for (int m=0; m<size<1>(tile_o); ++m)
                    CUTE_UNROLL
                    for (int a=0; a<size<0>(tile_o); ++a)
                        tOrO_float(a,m,dc*size<2>(tile_o)+n) = __fmaf_rn(row_alpha[a/2], tOrO_float(a,m,dc*size<2>(tile_o)+n), tile_o(a,m,n));
        }


    }

    } else {  // Even-tile consumer path
    CUTE_NO_UNROLL
    for (int n_block=n_block_max-1; n_block>=n_block_min; --n_block,++generation) {
        role_barrier<2,256>();
        CUTE_UNROLL
        for (int vector=0; vector<2; ++vector) {
            load_probability4(sP+(generation&1)*1024+vector*512+logical_tid*4,
                p_words(4*vector), p_words(4*vector+1),
                p_words(4*vector+2), p_words(4*vector+3));
        }
        float row_alpha[2] = {sAlpha[(generation&1)*64+thread_row],sAlpha[(generation&1)*64+thread_row+8]};
        role_barrier<3,128>();  // consumers only; producer may advance after READY
        auto sVt = make_tensor(make_smem_ptr(smem_half+8192+(generation&1)*8192),
                              typename Kernel_traits::SmemLayoutVTransposed{});
        constexpr int PV_COLS = 32;
        // Split the128-bit V copy atom into two fixed four-vector subsets.
        // Tile0 is a type/layout prototype only; no prototype data is read.
        typename Kernel_traits::GmemTiledCopyV future_v_copy;
        auto future_v_thread = future_v_copy.get_slice(logical_tid);
        auto future_v_prototype = local_tile(gV(_,_,0),
            Shape<Int<kBlockN>,Int<128>>{},make_coord(0,Int<0>{}));
        auto future_v_carry = make_fragment_like(future_v_thread.partition_S(future_v_prototype));
        static_assert(decltype(size(future_v_carry))::value == 32);
        static_assert(decltype(size(recast<uint32_t>(future_v_carry)))::value == 16);
        const bool future_v_enabled = n_block > n_block_min && generation+1 >= n_masking_steps;
        // READY(g) retired every consumer's PV(g-1). Opposite slot g+1 is free.
        auto future_v_store = [&](auto phase) {
            auto future_v_shared = make_tensor(
                make_smem_ptr(smem_half+8192+((generation+1)&1)*8192),
                typename Kernel_traits::SmemLayoutV{});
            auto future_v_half = local_tile(future_v_shared,
                Shape<Int<kBlockN>,Int<128>>{},make_coord(0,phase));
            auto future_v_destination = future_v_thread.partition_D(future_v_half);
            copy(future_v_copy,future_v_carry,future_v_destination);
        };
        auto future_v_load = [&](auto phase) {
            // Called only under the full-main future guard: n_block-1 is valid.
            auto future_v_half = local_tile(gV(_,_,n_block-1),
                Shape<Int<kBlockN>,Int<128>>{},make_coord(0,phase));
            auto future_v_source = future_v_thread.partition_S(future_v_half);
            copy(future_v_copy,future_v_source,future_v_carry);
        };
        // One written PV body; DC is a compile-time coordinate, never dynamic O indexing.
        for_each(make_seq<kHeadDim/PV_COLS>{},[&](auto dc_constant) {
            constexpr int dc = decltype(dc_constant)::value;
            if constexpr(dc == 0) {
                if (future_v_enabled) future_v_load(Int<0>{});
            }
            auto v_cols = local_tile(sVt, Shape<Int<PV_COLS>,Int<kBlockN>>{}, make_coord(dc,0));
            auto v_part = thr_mma_O.partition_B(v_cols);
            auto v_frag = thr_mma_O.make_fragment_B(v_part);
            auto v_src = s2r_thr_copy_V.partition_S(v_cols);
            auto v_dst = s2r_thr_copy_V.retile_D(v_frag);
            auto tile_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>,Int<PV_COLS>>{});
            // F01: same V load/fragment ownership; only high PV remains.
            CUTE_UNROLL
            for (int pb=0; pb<size<2>(v_part); ++pb) {
                copy(s2r_tiled_copy_V, v_src(_,_,pb), v_dst(_,_,pb));
            }
            clear(tile_o);
            CUTE_UNROLL
            for (int pb=0; pb<size<2>(v_part); ++pb) {
                gemm(tiled_mma, tOrP(_,_,pb), v_frag(_,_,pb), tile_o);
            }
            CUTE_UNROLL
            for (int n=0; n<size<2>(tile_o); ++n)
                CUTE_UNROLL
                for (int m=0; m<size<1>(tile_o); ++m)
                    CUTE_UNROLL
                    for (int a=0; a<size<0>(tile_o); ++a)
                        tOrO_float(a,m,dc*size<2>(tile_o)+n) = __fmaf_rn(row_alpha[a/2], tOrO_float(a,m,dc*size<2>(tile_o)+n), tile_o(a,m,n));

            if constexpr(dc == 3) {
                if (future_v_enabled) {
                    future_v_store(Int<0>{});
                    future_v_load(Int<1>{});
                }
            }
            if constexpr(dc == 7) {
                if (future_v_enabled) future_v_store(Int<1>{});
            }
        });


    }

    }  // End consumer paths
    role_barrier<4,256>();  // final PV retired before shared O alias
    float rL[2] = {sFinalL[thread_row],sFinalL[thread_row+8]};
    for (int i =0; i<2; i++) {
        // Sometimes the whole row of q might get masked out, for example where seqlen_q > seqlen_k,
        // in which case rL will be zero.
        if (rL[i] != 0.0f) {
            for (int j=0; j < tOrO_float(make_coord(_,i),_,_).size(); j++) {
                tOrO_float(make_coord(_,i),_,_)[j] /= rL[i];
            }
        } else {
            for (int j=0; j < tOrO_float(make_coord(_,i),_,_).size(); j++) {
                tOrO_float(make_coord(_,i),_,_)[j] = 0;
            }
        }

    }



    auto sO = make_tensor(make_smem_ptr(smem_half),typename Kernel_traits::SmemLayoutQ{});
    auto tOsO = thr_mma_O.partition_C(sO);
    auto tOrO = convert_type<half_t>(tOrO_float);
    copy(tOrO,tOsO);
    role_barrier<5,128>();  // original four consumer warps publish full D256 O
    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto thr_copy_O = gmem_tiled_copy_O.get_slice(logical_tid);
    auto tOsO_copy = thr_copy_O.partition_S(sO);
    auto tOgO_copy = thr_copy_O.partition_D(gO);
    auto cO_identity = make_identity_tensor(make_shape(Int<kBlockM>{},Int<kHeadDim>{}));
    auto tCqO = thr_copy_O.partition_S(cO_identity);
    masked_copy<Is_even_MN>(gmem_tiled_copy_O,tOsO_copy,tOgO_copy,tCqO,
                           seqlen_q-m_block*kBlockM,/*clear_D=*/false);

}

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN>
inline __device__ void compute_attn(half_t* __restrict__ q,
                                      half_t* __restrict__ k,
                                      half_t* __restrict__ v,
                                      half_t* __restrict__ o,
                                      float* __restrict__ l,
                                      int* __restrict__ cu_seqlens_q,
                                      int* __restrict__ cu_seqlens_k,
                                      int batch_size,
                                      int seqlen_q,
                                      int seqlen_k,
                                      int num_heads,
                                      int num_heads_k,
                                      int h_h_k_ratio,
                                      int head_dim,
                                      float softmax_scale,
                                      int is_casual) {
    const int m_block = blockIdx.y;
    // The block index for the batch.
    const int bidb = blockIdx.z;
    // The block index for the head.
    const int bidh = blockIdx.x;

    compute_attn_1rowblock<Kernel_traits, Is_causal, Is_even_MN>(q,
                                                    k,
                                                    v,
                                                    o,
                                                    l,
                                                    cu_seqlens_q,
                                                    cu_seqlens_k,
                                                    batch_size,
                                                    seqlen_q,
                                                    seqlen_k,
                                                    num_heads,
                                                    num_heads_k,
                                                    h_h_k_ratio,
                                                    head_dim,
                                                    softmax_scale,
                                                    is_casual,
                                                    bidb,
                                                    bidh,
                                                    m_block);
}

}  // namespace flash_attn_d256
