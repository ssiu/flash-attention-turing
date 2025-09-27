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
#include "kernel_traits.h"
#include "utils.h"
#include "mask.h"

using namespace cute;


// for some reason changing this into params struct is 15% slower for hdim = 128
template <typename Kernel_traits, bool Is_causal>
inline __device__ void compute_attn_1rowblock(half_t* __restrict__ q,
                          half_t* __restrict__ k,
                          half_t* __restrict__ v,
                          half_t* __restrict__ o,
                          float* __restrict__ l,
                          int batch_size,
                          int seqlen_q,
                          int seqlen_k,
                          int num_heads,
                          int head_dim,
                          int is_casual,
                          int bidb,
                          int bidh,
                          int m_block)
{

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    const BlockInfo binfo(seqlen_q, seqlen_k, bidb);


    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<half_t*>(q) + binfo.q_offset(seqlen_q * num_heads * head_dim, bidb)),
                            make_shape(seqlen_q, num_heads, head_dim),
                            make_stride(num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));




    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<half_t*>(k) + binfo.k_offset(seqlen_k * num_heads * head_dim, bidb)),
                            make_shape(seqlen_k, num_heads, head_dim),
                            make_stride(num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gK = local_tile(mK(_, bidh, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));




    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<half_t*>(v) + binfo.k_offset(seqlen_k * num_heads * head_dim, bidb)),
                            make_shape(head_dim, num_heads, seqlen_k),
                            make_stride(Int<1>{}, head_dim, num_heads * head_dim));

    Tensor gV = local_tile(mV(_, bidh, _), Shape<Int<kHeadDim>, Int<kBlockN>>{},
                           make_coord(0, _));



    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<half_t*>(o) + binfo.q_offset(seqlen_q * num_heads * head_dim, bidb)),
                            make_shape(seqlen_q, num_heads, head_dim),
                            make_stride(num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));

    // L = m + log l
    Tensor mL = make_tensor(make_gmem_ptr(reinterpret_cast<float*>(l)),
                             make_shape(batch_size, num_heads, seqlen_q),
                             make_stride(seqlen_q * num_heads, seqlen_q, Int<1>{}));

    Tensor gL = local_tile(mL(bidb, bidh, _), Shape<Int<kBlockM>>{},
                           make_coord(m_block));

//     if (thread0()){
//         print(gL);
//         printf("\n");
//     }


    extern __shared__ char smem_[];


    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + kBlockM*kHeadDim, typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutV{});
    Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), typename Kernel_traits::SmemLayoutQ{});

//     if (thread0()) {
//         print(gQ);
//         print("\n");
//         print(sQ);
//         print("\n");
//     }

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int thread_row = warp_id * 16 + lane_id / 4;

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

    // gmem -> smem for Q, K, V
    typename Kernel_traits::GmemTiledCopyQK gmem_tiled_copy_QK;
    typename Kernel_traits::GmemTiledCopyV gmem_tiled_copy_V;
    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;

    ThrCopy thr_copy_QK = gmem_tiled_copy_QK.get_slice(threadIdx.x);
    Tensor tQgQ = thr_copy_QK.partition_S(gQ);
    Tensor tQsQ = thr_copy_QK.partition_D(sQ);

    Tensor tKgK = thr_copy_QK.partition_S(gK);
    Tensor tKsK = thr_copy_QK.partition_D(sK);
    Tensor tKrK = make_fragment_like(tKsK);


    ThrCopy thr_copy_V = gmem_tiled_copy_V.get_slice(threadIdx.x);
    Tensor tVgV = thr_copy_V.partition_S(gV);
    Tensor tVsV = thr_copy_V.partition_D(sV);
    Tensor tVrV = make_fragment_like(tVsV);

    // smem -> gmem for O
    ThrCopy thr_copy_O = gmem_tiled_copy_O.get_slice(threadIdx.x);
    Tensor tOsO_copy = thr_copy_O.partition_S(sO);
    Tensor tOgO_copy = thr_copy_O.partition_D(gO);


    typename Kernel_traits::TiledMma tiled_mma;

    // mma for S = QK^T
    ThrMMA thr_mma_S = tiled_mma.get_slice(threadIdx.x);
    Tensor tSsQ = thr_mma_S.partition_A(sQ);
    Tensor tSsK = thr_mma_S.partition_B(sK);

    Tensor tSrQ = thr_mma_S.make_fragment_A(tSsQ);
    Tensor tSrK = thr_mma_S.make_fragment_B(tSsK);
    Tensor tSrS_float = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});


    // mma for O = PV
    ThrMMA thr_mma_O = tiled_mma.get_slice(threadIdx.x);
    Tensor tOsV = thr_mma_O.partition_B(sV);
    Tensor tOrV = thr_mma_O.make_fragment_B(tOsV);
    Tensor tOrO_float = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor tOsO = thr_mma_O.partition_C(sO);


    //  each warp only process 16 rows
    auto s2r_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQ{}, tiled_mma);
    auto s2r_thr_copy_Q = s2r_tiled_copy_Q.get_slice(threadIdx.x);
    auto tSsQ_copy_view = s2r_thr_copy_Q.partition_S(sQ);
    auto tSrQ_copy_view = s2r_thr_copy_Q.retile_D(tSrQ);

    auto s2r_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomK{}, tiled_mma);
    auto s2r_thr_copy_K = s2r_tiled_copy_K.get_slice(threadIdx.x);
    auto tSsK_copy_view = s2r_thr_copy_K.partition_S(sK);
    auto tSrK_copy_view = s2r_thr_copy_K.retile_D(tSrK);

    auto s2r_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomV{}, tiled_mma);
    auto s2r_thr_copy_V = s2r_tiled_copy_V.get_slice(threadIdx.x);
    auto tOsV_copy_view = s2r_thr_copy_V.partition_S(sV);
    auto tOrV_copy_view = s2r_thr_copy_V.retile_D(tOrV);



    // int KV_TILE_MAX = 0;
    // int KV_TILE_NO_MASK = 0;
    // int KV_TILE_MASK_START = 0;
    // int KV_TILE_MASK_END = 0;
    // int KV_TILE_SHIFT = 0;
    // // 0 -> KV_TILE_NO_MASK = KV_TILE_MASK_START -> KV_TILE_MASK_END
    // if constexpr (Is_causal) {
    //     // because we parallelize across Q_TILE,
    //     // we need the blocks that do not need a mask
    //     // and then compute the blocks that need masking
    //     if (seqlen_q <= seqlen_k) {
    //         // number of KV_TILES that does not need a mask
    //         //KV_TILE_NO_MASK = (m_block + seqlen_k - seqlen_q) * kBlockM / kBlockN;
    //         KV_TILE_NO_MASK = (m_block + (seqlen_k - seqlen_q) / kBlockM) * kBlockM / kBlockN;
    //         KV_TILE_MASK_START = KV_TILE_NO_MASK;
    //         KV_TILE_MASK_END = KV_TILE_NO_MASK + (kBlockM / kBlockN);
    //         KV_TILE_MAX = KV_TILE_NO_MASK;
    //     } else {
    //         // number of KV_TILES that does not need a mask
    //         KV_TILE_SHIFT = (m_block - (seqlen_q - seqlen_k) / kBlockM) * kBlockM / kBlockN;
    //         KV_TILE_NO_MASK = max(KV_TILE_SHIFT, 0);
    //         KV_TILE_MASK_START = KV_TILE_NO_MASK;
    //         //KV_TILE_MASK_END = KV_TILE_NO_MASK + (kBlockM / kBlockN);
    //         KV_TILE_MASK_END = KV_TILE_SHIFT < 0 ? 0 : KV_TILE_MASK_START + (kBlockM / kBlockN);
    //         KV_TILE_MAX = KV_TILE_NO_MASK;
    //     }

    // } else {
    //     KV_TILE_MAX = size<3>(tKgK);
    // }


    const int n_block_min = 0; 
    int n_block_max = ceil_div(seqlen_k, kBlockN);
    if constexpr(Is_causal) {
        n_block_max = (m_block + 1 + (seqlen_k - seqlen_q) / kBlockM) * kBlockM / kBlockN;
    }        
    
    const int masking_steps = (!is_casual) ? 1 : ceil_div(kBlockM, kBlockN); 
    int n_block_no_mask = n_block_max - masking_steps;

    // if seqlen_q > seqlen_k we exit early for the blocks with rows that are fully masked
    if (n_block_max < 0) {return;}

    auto QK_BLOCK_MAX = size<2>(tSsK);
    auto PV_BLOCK_MAX = size<2>(tOsV);

    // prologue

    copy(gmem_tiled_copy_QK, tQgQ, tQsQ);
    copy(gmem_tiled_copy_QK, tKgK(_,_,_,0), tKrK);
    //copy(gmem_tiled_copy_V, tVgV(_,_,_,0), tVrV);

    clear(tOrO_float);

    // main loop
    CUTE_NO_UNROLL
    for (int n_block = 0; n_block < n_block_no_mask; ++n_block) {


        copy(gmem_tiled_copy_QK, tKrK, tKsK);
        //copy(gmem_tiled_copy_V, tVrV, tVsV);

        __syncthreads();

        clear(tSrS_float);

        if (n_block + 1 < n_block_no_mask) {
            copy(gmem_tiled_copy_QK, tKgK(_,_,_,n_block + 1), tKrK);
            //copy(gmem_tiled_copy_V, tVgV(_,_,_,kv_tile + 1), tVrV);
        }

        CUTE_UNROLL
        for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
            copy(s2r_tiled_copy_Q, tSsQ_copy_view(_,_,qk_block), tSrQ_copy_view(_,_,qk_block));
            copy(s2r_tiled_copy_K, tSsK_copy_view(_,_,qk_block), tSrK_copy_view(_,_,qk_block));

            gemm(tiled_mma, tSrQ(_,_,qk_block), tSrK(_,_,qk_block), tSrS_float);

        }

        __syncthreads();
        copy(gmem_tiled_copy_V, tVgV(_,_,_,n_block), tVsV);
        __syncthreads();

//         if (thread0) {
//             print_tensor(tSrS_float);
//         }

        for (int i=0;i< tSrS_float.size();i ++ ) {
            tSrS_float[i] *= 1.0f / sqrtf(kHeadDim);
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
               rM[i] = fmaxf(rM[i], __shfl_down_sync(mask, rM[i], offset));
            }
        }


        // sync rM

        for (int i =0; i<2; i++) {
            rM[i] = __shfl_sync(mask, rM[i], lane_id_to_read_from);
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



        // rescale l and also reset rD to 0
//         for (int i =0; i<2; i++) {
//             //rL[i] = expf(rM_old[i] - rM[i]) * rL_old[i];
//             rL[i] = exp2f(rM_old[i] * float(M_LOG2E) - max_scaled) * rL_old[i];
//             rD[i] = 0.0f;
//         }

        // if (thread0()){
        //     printf("kv_tile = %d, rL for this loop: %f\n", kv_tile, rL[0]);
        // }

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



        // can just keep the correct rL to lane 0
        for (int i =0; i<2; i++) {
            rL[i] += rD[i];
        }
        // if (thread0()){
        //     printf("kv_tile = %d, rL after adding rD: %f\n", kv_tile, rL[0]);
        // }


        // sync rL
        for (int i =0; i<2; i++) {
            rL[i] = __shfl_sync(mask, rL[i], lane_id_to_read_from);
        }



//         constexpr int num_element = decltype(size(tSrS_float))::value;
//
//         cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
//         auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tSrS_float.data()));
//
//         Tensor tOrP = make_tensor(make_rmem_ptr<half_t>(&frag), tSrS_float.layout());

        Tensor tOrP = convert_type<half_t>(tSrS_float);


        // rescale O

        for (int i =0; i<2; i++) {
            for (int j=0; j < tOrO_float(make_coord(_,i),_,_).size(); j++) {
                tOrO_float(make_coord(_,i),_,_)[j] = expf(rM_old[i] - rM[i]) * tOrO_float(make_coord(_,i),_,_)[j];
            }
        }



        CUTE_UNROLL
        for (int pv_block = 0; pv_block < PV_BLOCK_MAX; pv_block++) {
            copy(s2r_tiled_copy_V, tOsV_copy_view(_,_,pv_block), tOrV_copy_view(_,_,pv_block));

            gemm(tiled_mma, tOrP(_,_,pv_block), tOrV(_,_,pv_block), tOrO_float);

        }



        // update m and l
        for (int i = 0; i< 2;i++) {
            rM_old[i] = rM[i];
            rL_old[i] = rL[i];
        }


        __syncthreads();

    }
    // end of KV loop



    // these are the blocks that need masking
    Mask<Is_causal> accum_s_mask(seqlen_q, seqlen_k);
    // constexpr bool Is_even_MN = true; 

    

    copy(gmem_tiled_copy_QK, tKgK(_,_,_,n_block_no_mask), tKrK);
    for (int n_block = n_block_no_mask; n_block < n_block_max; ++n_block) {

        copy(gmem_tiled_copy_QK, tKrK, tKsK);

        __syncthreads();

        clear(tSrS_float);

        if (n_block + 1 < n_block_max) {
            copy(gmem_tiled_copy_QK, tKgK(_,_,_,n_block + 1), tKrK);
        }

        CUTE_UNROLL
        for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
            copy(s2r_tiled_copy_Q, tSsQ_copy_view(_,_,qk_block), tSrQ_copy_view(_,_,qk_block));
            copy(s2r_tiled_copy_K, tSsK_copy_view(_,_,qk_block), tSrK_copy_view(_,_,qk_block));

            gemm(tiled_mma, tSrQ(_,_,qk_block), tSrK(_,_,qk_block), tSrS_float);

        }

        __syncthreads();
        copy(gmem_tiled_copy_V, tVgV(_,_,_,n_block), tVsV);
        __syncthreads();


        // for now we rescale before we apply causal mask
        for (int i=0;i< tSrS_float.size();i ++ ) {
            tSrS_float[i] *= 1.0f / sqrtf(kHeadDim);
        }



        accum_s_mask.template apply_mask_fwd<Is_causal>(
            tSrS_float, warp_id, lane_id, n_block, n_block_max, kBlockN
        );

        if (seqlen_q == 128 && seqlen_k == 128 && thread(0)) {
            print(tSrS_float);
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
                tSrS_float(make_coord(_,i),_,_)[j] = expf(tSrS_float(make_coord(_,i),_,_)[j] - rM[i]);
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



        // can just keep the correct rL to lane 0
        for (int i =0; i<2; i++) {
            rL[i] += rD[i];
        }

        // if (thread0()){
        //     printf("kv_tile = %d, rL after adding rD: %f\n", kv_tile, rL[0]);
        // }


        // sync rL
        for (int i =0; i<2; i++) {
            rL[i] = __shfl_sync(mask, rL[i], lane_id_to_read_from);
        }



//             constexpr int num_element = decltype(size(tSrS_float))::value;
//
//             cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
//             auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tSrS_float.data()));
//
//             Tensor tOrP = make_tensor(make_rmem_ptr<half_t>(&frag), tSrS_float.layout());
        Tensor tOrP = convert_type<half_t>(tSrS_float);


        // rescale O

        for (int i =0; i<2; i++) {
            for (int j=0; j < tOrO_float(make_coord(_,i),_,_).size(); j++) {
                tOrO_float(make_coord(_,i),_,_)[j] = expf(rM_old[i] - rM[i]) * tOrO_float(make_coord(_,i),_,_)[j];
            }
        }



        CUTE_UNROLL
        for (int pv_block = 0; pv_block < PV_BLOCK_MAX; pv_block++) {
            copy(s2r_tiled_copy_V, tOsV_copy_view(_,_,pv_block), tOrV_copy_view(_,_,pv_block));

            gemm(tiled_mma, tOrP(_,_,pv_block), tOrV(_,_,pv_block), tOrO_float);

        }

        // update m and l
        for (int i = 0; i< 2;i++) {
            rM_old[i] = rM[i];
            rL_old[i] = rL[i];
        }

        __syncthreads();

    }
    


    gL[thread_row] = rM[0] + logf(rL[0]);
    gL[thread_row + 8] = rM[1] + logf(rL[1]);



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


//     constexpr int num_element = decltype(size(tOrO_float))::value;
//
//     cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
//     auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tOrO_float.data()));
//     Tensor tOrO = make_tensor(make_rmem_ptr<half_t>(&frag), tOrO_float.layout());

    Tensor tOrO = convert_type<half_t>(tOrO_float);

    copy(tOrO, tOsO);

    __syncthreads();

    copy(gmem_tiled_copy_O, tOsO_copy, tOgO_copy);


}


template<typename Kernel_traits, bool Is_causal>
inline __device__ void compute_attn(half_t* __restrict__ q,
                                      half_t* __restrict__ k,
                                      half_t* __restrict__ v,
                                      half_t* __restrict__ o,
                                      float* __restrict__ l,
                                      int batch_size,
                                      int seqlen_q,
                                      int seqlen_k,
                                      int num_heads,
                                      int head_dim,
                                      int is_casual) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    compute_attn_1rowblock<Kernel_traits, Is_causal>(q,
                                                    k,
                                                    v,
                                                    o,
                                                    l,
                                                    batch_size,
                                                    seqlen_q,
                                                    seqlen_k,
                                                    num_heads,
                                                    head_dim,
                                                    is_casual,
                                                    bidb,
                                                    bidh,
                                                    m_block);
}