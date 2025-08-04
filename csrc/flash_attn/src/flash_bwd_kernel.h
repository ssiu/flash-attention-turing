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
#include "block_info.h"

using namespace cute;

// kBlockN = 64 works
// kBlockM = 64 doesnt work
// #define K_BLOCK_M 64
// #define K_BLOCK_N 64



//// each block handles kBlockM x headdim
//// use mma_dq to partition the block, and convert from float to half
//template <typename Kernel_traits, bool Is_causal>
//inline __device__ void convert_dq(float* dq_float_ptr,
//              half_t* dq_ptr,
//              int batch_size, int seq_len, int num_heads, int head_dim, int is_causal)
//{
//
//        constexpr int kBlockM = Kernel_traits::kBlockM;
//        constexpr int kBlockN = Kernel_traits::kBlockN;
//        constexpr int kHeadDim = Kernel_traits::kHeadDim;
//        Tensor mdQ_float = make_tensor(make_gmem_ptr(dq_float_ptr),
//                           make_shape(batch_size, seq_len, num_heads, head_dim),
//                           make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));
//
//        Tensor gdQ_float = local_tile(mdQ_float(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                          make_coord(blockIdx.z, 0));
//
//
//        Tensor mdQ = make_tensor(make_gmem_ptr(dq_ptr),
//                           make_shape(batch_size, seq_len, num_heads, head_dim),
//                           make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));
//
//        Tensor gdQ = local_tile(mdQ(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                          make_coord(blockIdx.z, 0));
//
//
//        typename Kernel_traits::TiledMma_dQ tiled_mma_dQ;
//        ThrMMA thr_mma_dQ = tiled_mma_dQ.get_slice(threadIdx.x);
//
//        Tensor tdQgdQ_float = thr_mma_dQ.partition_C(gdQ_float);
//        Tensor tdQgdQ = thr_mma_dQ.partition_C(gdQ);
//
//        Tensor tdQrdQ_float = thr_mma_dQ.make_fragment_C(tdQgdQ_float);
//
//        copy(tdQgdQ_float, tdQrdQ_float);
//
//        Tensor tdQrdQ = convert_type<half_t>(tdQrdQ_float);
//
//        copy(tdQrdQ, tdQgdQ);
//
//
//}




template <typename Kernel_traits, bool Is_causal>
inline __device__ void compute_dq_1rowblock(
    half_t * __restrict__ q_ptr,
    half_t * __restrict__ k_ptr,
    half_t * __restrict__ v_ptr,
    float *__restrict__ l_ptr,
    float * __restrict__ d_ptr,
    half_t * __restrict__ do_ptr,
    half_t* __restrict__ dq_ptr,
    int batch_size, int seq_len, int num_heads, int head_dim, int is_causal,
    int bidb,
      int bidh,
      int m_block
)
{
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    //constexpr int kNWarps = 8;
    //constexpr int kNThreads = kNWarps * 32;

    const BlockInfo binfo(seq_len, bidb);

    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;

    // for 8 warps, the 32x32 tiled mmas is like
    // -------------------------------------
    // | Warp 0 | Warp 2 | Warp 4 | Warp 6 |
    // -------------------------------------
    // | Warp 1 | Warp 3 | Warp 5 | Warp 7 |
    // -------------------------------------
    // for 64 x 64 tiledmma, each thread computes 16 numbers and each row can be accessed by
    // print(tc);
    // print_tensor(tc);
    // ptr[32b](0x7ea2408d8910) o ((_2,_2),_2,_2):((_1,_512),_2048,_32)
    // for (int i=0;i<2;i++) {
    //     for (int j=0;j<2;j++) {
    //         print_tensor(tc(make_coord(_,j),i,_));
    //     }
    // }



    // Q
//    Tensor mQ = make_tensor(make_gmem_ptr(q_ptr),
//                            make_shape(batch_size, seq_len, num_heads, head_dim),
//                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));
//
//    Tensor gQ = local_tile(mQ(bidb, _, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                           make_coord(m_block, 0));
    Tensor mQ = make_tensor(make_gmem_ptr(q_ptr + binfo.q_offset(seq_len * num_heads * head_dim, bidb)),
                            make_shape(seq_len, num_heads, head_dim),
                            make_stride(num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));

    // K
//    Tensor mK = make_tensor(make_gmem_ptr(k_ptr),
//                            make_shape(batch_size, seq_len, num_heads, head_dim),
//                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));
//
//    Tensor gK = local_tile(mK(bidb, _, bidh, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                           make_coord(_, 0));

    Tensor mK = make_tensor(make_gmem_ptr(k_ptr + binfo.q_offset(seq_len * num_heads * head_dim, bidb)),
                            make_shape(seq_len, num_heads, head_dim),
                            make_stride(num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gK = local_tile(mK(_, bidh, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));

    // V
//    Tensor mV = make_tensor(make_gmem_ptr(v_ptr),
//                            make_shape(batch_size, seq_len, num_heads, head_dim),
//                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));
//
//    Tensor gV = local_tile(mV(bidb, _, bidh, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                           make_coord(_, 0));

    Tensor mV = make_tensor(make_gmem_ptr(v_ptr + binfo.q_offset(seq_len * num_heads * head_dim, bidb)),
                            make_shape(seq_len, num_heads, head_dim),
                            make_stride(num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gV = local_tile(mV(_, bidh, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));


    // L = m + log l
    Tensor mL = make_tensor(make_gmem_ptr(l_ptr),
                             make_shape(batch_size, num_heads, seq_len),
                             make_stride(seq_len * num_heads,  seq_len, Int<1>{}));

    Tensor gL = local_tile(mL(bidb, bidh, _), Shape<Int<kBlockM>>{},
                           make_coord(m_block));


    Tensor mD = make_tensor(make_gmem_ptr(d_ptr),
                             make_shape(batch_size, num_heads, seq_len),
                             make_stride(seq_len * num_heads,  seq_len, Int<1>{}));

    Tensor gD = local_tile(mD(bidb, bidh, _), Shape<Int<kBlockM>>{},
                           make_coord(m_block));

    // dO
//    Tensor mdO = make_tensor(make_gmem_ptr(do_ptr ),
//                             make_shape(batch_size, seq_len, num_heads, head_dim),
//                             make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));
//
//    Tensor gdO = local_tile(mdO(bidb, _, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                           make_coord(m_block, 0));

    Tensor mdO = make_tensor(make_gmem_ptr(do_ptr + binfo.q_offset(seq_len * num_heads * head_dim, bidb)),
                             make_shape(seq_len, num_heads, head_dim),
                             make_stride(num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gdO = local_tile(mdO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));


    // dQ
//    Tensor mdQ = make_tensor(make_gmem_ptr(dq_ptr),
//                            make_shape(batch_size, seq_len, num_heads, head_dim),
//                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));
//
//    Tensor gdQ = local_tile(mdQ(bidb, _, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                           make_coord(m_block, 0));

    Tensor mdQ = make_tensor(make_gmem_ptr(dq_ptr + binfo.q_offset(seq_len * num_heads * head_dim, bidb)),
                            make_shape(seq_len, num_heads, head_dim),
                            make_stride(num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gdQ = local_tile(mdQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));
    extern __shared__ char smem_[];


    // 64 * 128 = 16KB
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), typename Kernel_traits::SmemLayoutQ{});
    Tensor sQt = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), typename Kernel_traits::SmemLayoutQTransposed{});

    // 64 * 128 = 16KB
    Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKVTransposed{});

    // 64 * 128 = 16KB
    Tensor sdO = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutQ{});
    Tensor sdOt = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutQTransposed{});

    // 64 * 128 = 16KB
    Tensor sV = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});

    // 64 * 64 = 8KB
    Tensor sP = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sPt = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutPdSTransposed{});

    // 64 * 64 = 8KB
    Tensor sdS = make_tensor(sP.data() + size(sP), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sdSt = make_tensor(sP.data() + size(sP), typename Kernel_traits::SmemLayoutPdSTransposed{});

    Tensor sdQ = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), typename Kernel_traits::SmemLayoutQ{});


    //int thread_id = threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    //int thread_row = warp_id * 16 + lane_id / 4;
    int warp_offset = (warp_id % 2) * 16;
    int thread_offset = lane_id / 4;

    float rL[2][2] = {0};
    float rD[2][2] = {0};

    // Copy operation
    //GmemTiledCopyQKV gmem_tiled_copy_QKV;

    typename Kernel_traits::GmemTiledCopy gmem_tiled_copy_QKV;

    ThrCopy thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(threadIdx.x);

    Tensor tQgQ = thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = thr_copy_QKV.partition_D(sQ);

    Tensor tKgK = thr_copy_QKV.partition_S(gK);
    Tensor tKsK = thr_copy_QKV.partition_D(sK);
    Tensor tKrK = make_fragment_like(tKsK);

    Tensor tVgV = thr_copy_QKV.partition_S(gV);
    Tensor tVsV = thr_copy_QKV.partition_D(sV);
    Tensor tVrV = make_fragment_like(tVsV);

    Tensor tdOgdO = thr_copy_QKV.partition_S(gdO);
    Tensor tdOsdO = thr_copy_QKV.partition_D(sdO);

    Tensor tdQsdQ_copy = thr_copy_QKV.partition_S(sdQ);
    Tensor tdQgdQ_copy = thr_copy_QKV.partition_D(gdQ);


    // S = QK^T
    typename Kernel_traits::TiledMma_SdP tiled_mma_S;
    ThrMMA thr_mma_S = tiled_mma_S.get_slice(threadIdx.x);
    Tensor tSgQ = thr_mma_S.partition_A(gQ);
    Tensor tSsQ = thr_mma_S.partition_A(sQ);
    Tensor tSrQ = thr_mma_S.make_fragment_A(tSsQ);

    Tensor tSgK = thr_mma_S.partition_B(gK);
    Tensor tSsK = thr_mma_S.partition_B(sK);
    Tensor tSrK = thr_mma_S.make_fragment_B(tSsK);

    Tensor tSrS_float = partition_fragment_C(tiled_mma_S, Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tSsP = thr_mma_S.partition_C(sP);


    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQdOPdS{}, tiled_mma_S);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(threadIdx.x);
    auto tSsQ_copy_view = smem_thr_copy_Q.partition_S(sQ);
    auto tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomKV{}, tiled_mma_S);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_slice(threadIdx.x);
    auto tSsK_copy_view = smem_thr_copy_K.partition_S(sK);
    auto tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);



    // dP = dOV^T
    typename Kernel_traits::TiledMma_SdP tiled_mma_dP;
    ThrMMA thr_mma_dP = tiled_mma_dP.get_slice(threadIdx.x);

    Tensor tdPgdO = thr_mma_dP.partition_A(gdO);
    Tensor tdPsdO = thr_mma_dP.partition_A(sdO);
    Tensor tdPrdO = thr_mma_dP.make_fragment_A(tdPsdO);

    Tensor tdPgV = thr_mma_dP.partition_B(gV);
    Tensor tdPsV = thr_mma_dP.partition_B(sV);
    Tensor tdPrV = thr_mma_dP.make_fragment_B(tdPsV);

    Tensor tdPrdP_float = partition_fragment_C(tiled_mma_dP, Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tdPsdS = thr_mma_dP.partition_C(sdS);

    auto smem_tiled_copy_dO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQdOPdS{}, tiled_mma_dP);
    auto smem_thr_copy_dO = smem_tiled_copy_dO.get_slice(threadIdx.x);
    auto tdPsdO_copy_view = smem_thr_copy_dO.partition_S(sdO);
    auto tdPrdO_copy_view = smem_thr_copy_dO.retile_D(tdPrdO);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomKV{}, tiled_mma_dP);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_slice(threadIdx.x);
    auto tdPsV_copy_view = smem_thr_copy_V.partition_S(sV);
    auto tdPrV_copy_view = smem_thr_copy_V.retile_D(tdPrV);


    // dQ = dSK
    typename Kernel_traits::TiledMma_dQ tiled_mma_dQ;
    ThrMMA thr_mma_dQ = tiled_mma_dQ.get_slice(threadIdx.x);
    Tensor tdQsdS = thr_mma_dQ.partition_A(sdS);
    Tensor tdQrdS = thr_mma_dQ.make_fragment_A(tdQsdS);
    Tensor tdQsKt = thr_mma_dQ.partition_B(sKt);
    Tensor tdQrKt = thr_mma_dQ.make_fragment_B(tdQsKt);

    Tensor tdQrdQ_float = partition_fragment_C(tiled_mma_dQ, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor tdQsdQ = thr_mma_dQ.partition_C(sdQ);
    Tensor tdQgdQ = thr_mma_dQ.partition_C(gdQ);

    auto smem_tiled_copy_dS = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQdOPdS{}, tiled_mma_dQ);
    auto smem_thr_copy_dS = smem_tiled_copy_dS.get_slice(threadIdx.x);
    auto tdQsdS_copy_view = smem_thr_copy_dS.partition_S(sdS);
    auto tdQrdS_copy_view = smem_thr_copy_dS.retile_D(tdQrdS);

    auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomKt{}, tiled_mma_dQ);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_slice(threadIdx.x);
    auto tdQsKt_copy_view = smem_thr_copy_Kt.partition_S(sKt);
    auto tdQrKt_copy_view = smem_thr_copy_Kt.retile_D(tdQrKt);

    int KV_TILE_MAX = 0;
    int KV_TILE_NO_MASK = 0;
    int KV_TILE_MASK_START = 0;
    int KV_TILE_MASK_END = 0;

    if constexpr (Is_causal) {
        // number of KV_TILES that does not need a mask
        KV_TILE_NO_MASK = m_block * kBlockM / kBlockN;
        KV_TILE_MASK_START = KV_TILE_NO_MASK;
        KV_TILE_MASK_END = KV_TILE_MASK_START + kBlockM / kBlockN;
        KV_TILE_MAX = KV_TILE_NO_MASK;
    } else {
        KV_TILE_MAX = size<3>(tSgK);
    }


    //auto KV_TILE_MAX = size<3>(tSgK);
    auto QK_BLOCK_MAX = size<2>(tSsK);
    auto dSKt_BLOCK_MAX = size<2>(tdQsdS);


    // load K, V, dK, dV tiles

    copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
    //copy(tSgK, tSsK);
    copy(gmem_tiled_copy_QKV, tdOgdO, tdOsdO);

    copy(gmem_tiled_copy_QKV, tKgK(_,_,_,0), tKrK);
    copy(gmem_tiled_copy_QKV, tVgV(_,_,_,0), tVrV);



    clear(tdQrdQ_float);

    CUTE_NO_UNROLL
    for (int kv_tile = 0; kv_tile < KV_TILE_MAX; ++kv_tile) {
        clear(tSrS_float);
        clear(tdPrdP_float);

        copy(gmem_tiled_copy_QKV, tKrK, tKsK);
        copy(gmem_tiled_copy_QKV, tVrV, tVsV);


        __syncthreads();


        if (kv_tile + 1 < KV_TILE_MAX) {
            copy(gmem_tiled_copy_QKV, tKgK(_,_,_,kv_tile + 1), tKrK);
            copy(gmem_tiled_copy_QKV, tVgV(_,_,_,kv_tile + 1), tVrV);
        }



        CUTE_UNROLL
        for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
            copy(smem_tiled_copy_Q, tSsQ_copy_view(_,_,qk_block), tSrQ_copy_view(_,_,qk_block));
            copy(smem_tiled_copy_K, tSsK_copy_view(_,_,qk_block), tSrK_copy_view(_,_,qk_block));
            copy(smem_tiled_copy_dO, tdPsdO_copy_view(_,_,qk_block), tdPrdO_copy_view(_,_,qk_block));
            copy(smem_tiled_copy_V, tdPsV_copy_view(_,_,qk_block), tdPrV_copy_view(_,_,qk_block));

            gemm(tiled_mma_S, tSrQ(_,_,qk_block), tSrK(_,_,qk_block), tSrS_float);
            gemm(tiled_mma_dP, tdPrdO(_,_,qk_block), tdPrV(_,_,qk_block), tdPrdP_float);
        }


        __syncthreads();


        // load rL, rD from gmem to rmem
        for (int i=0;i<2;i++) {
            for (int j=0;j<2;j++) {
                rL[i][j] = gL((warp_offset + thread_offset + 8 * j + 32 * i));
                rD[i][j] = gD((warp_offset + thread_offset + 8 * j + 32 * i));
            }
        }

        //printf("kv_tile = %d, thread = %d, rD[0] = %f\n", kv_tile, threadIdx.x, rD[0][0]);

        // rescale S
        for (int i=0;i< tSrS_float.size();i ++ ) {
            tSrS_float[i] *= 1.0f / sqrtf(kHeadDim);
        }

        Tensor tSrP_float = tSrS_float;
        Tensor tdPrdS_float = tdPrdP_float;


        //printf("kv_tile = %d, thread = %d, dP[0] = %f\n", kv_tile, threadIdx.x, tdPrdP_float[0]);





        // compute P = exp(S-l)
        for (int i=0; i<2; i++) {
            for (int j=0;j<2;j++) {
                for (int k=0; k< tSrS_float(make_coord(_,j),i,_).size(); k++) {
                    tSrP_float(make_coord(_,j),i,_)[k] = expf(tSrS_float(make_coord(_,j),i,_)[k] - rL[i][j]);
                }
            }
        }

        // compute dS = P \circ (dP - D)
        // tS has the same mma layout as tdP
        for (int i=0; i<2; i++) {
            for (int j=0;j<2;j++) {
                for (int k=0; k< tdPrdP_float(make_coord(_,j),i,_).size(); k++) {
                    tdPrdS_float(make_coord(_,j),i,_)[k] = tSrP_float(make_coord(_,j),i,_)[k] * (tdPrdP_float(make_coord(_,j),i,_)[k] - rD[i][j]);
                }
            }
        }

        // convert dS from fp32 to fp16
//         constexpr int num_element_dS = decltype(size(tdPrdS_float))::value;
//
//         cutlass::NumericArrayConverter<half_t, float, num_element_dS> convert_op_dS;
//         auto frag_dS = convert_op_dS(*reinterpret_cast<const cutlass::Array<float, num_element_dS> *>(tdPrdS_float.data()));
//
//         Tensor tdPrdS = make_tensor(make_rmem_ptr<half_t>(&frag_dS), tdPrdS_float.layout());

        Tensor tdPrdS = convert_type<half_t>(tdPrdS_float);

        //copy(tSrP, tSsP);
        copy(tdPrdS, tdPsdS);


        __syncthreads();
//         if (thread(0) && kv_tile == 0) {
//
//             print_tensor(tdPsdS(_,0));
//         }

        // dQ += dSK

        CUTE_UNROLL
        for (int dskt_block = 0; dskt_block < dSKt_BLOCK_MAX; dskt_block++) {
            copy(smem_tiled_copy_dS, tdQsdS_copy_view(_,_,dskt_block), tdQrdS_copy_view(_,_,dskt_block));
            copy(smem_tiled_copy_Kt, tdQsKt_copy_view(_,_,dskt_block), tdQrKt_copy_view(_,_,dskt_block));


            gemm(tiled_mma_dQ, tdQrdS(_,_,dskt_block), tdQrKt(_,_,dskt_block), tdQrdQ_float);

        }

        __syncthreads();

    }


    if (Is_causal) {

        copy(gmem_tiled_copy_QKV, tKgK(_,_,_,KV_TILE_MASK_START), tKrK);
        copy(gmem_tiled_copy_QKV, tVgV(_,_,_,KV_TILE_MASK_START), tVrV);
        CUTE_NO_UNROLL
        for (int kv_tile = KV_TILE_MASK_START; kv_tile < KV_TILE_MASK_END; ++kv_tile) {
            clear(tSrS_float);
            clear(tdPrdP_float);

            copy(gmem_tiled_copy_QKV, tKrK, tKsK);
            copy(gmem_tiled_copy_QKV, tVrV, tVsV);


            __syncthreads();


            if (kv_tile + 1 < KV_TILE_MASK_END) {
                copy(gmem_tiled_copy_QKV, tKgK(_,_,_,kv_tile + 1), tKrK);
                copy(gmem_tiled_copy_QKV, tVgV(_,_,_,kv_tile + 1), tVrV);
            }



            CUTE_UNROLL
            for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
                copy(smem_tiled_copy_Q, tSsQ_copy_view(_,_,qk_block), tSrQ_copy_view(_,_,qk_block));
                copy(smem_tiled_copy_K, tSsK_copy_view(_,_,qk_block), tSrK_copy_view(_,_,qk_block));
                copy(smem_tiled_copy_dO, tdPsdO_copy_view(_,_,qk_block), tdPrdO_copy_view(_,_,qk_block));
                copy(smem_tiled_copy_V, tdPsV_copy_view(_,_,qk_block), tdPrV_copy_view(_,_,qk_block));

                gemm(tiled_mma_S, tSrQ(_,_,qk_block), tSrK(_,_,qk_block), tSrS_float);
                gemm(tiled_mma_dP, tdPrdO(_,_,qk_block), tdPrV(_,_,qk_block), tdPrdP_float);
            }


//             if (blockIdx.z ==0 && warp_id % 2 ==0 && lane_id < 4 && kv_tile == 0) {
//                 printf("kv_tile = %d, warp_id = %d, lane_id = %d\n", kv_tile, warp_id, lane_id);
//                 print_tensor(tSrS_float(make_coord(_,_), 0, _ ));
//
//             }


            __syncthreads();

            // load rL, rD from gmem to rmem
            for (int i=0;i<2;i++) {
                for (int j=0;j<2;j++) {
                    rL[i][j] = gL((warp_offset + thread_offset + 8 * j + 32 * i));
                    rD[i][j] = gD((warp_offset + thread_offset + 8 * j + 32 * i));
                }
            }

            // rescale S
            for (int i=0;i< tSrS_float.size();i ++ ) {
                tSrS_float[i] *= 1.0f / sqrtf(kHeadDim);
            }

//             if (blockIdx.z ==0 && warp_id == 2 && lane_id == 0 && kv_tile == 0) {
//                 printf("kv_tile = %d, warp_id = %d, lane_id = %d\n", kv_tile, warp_id, lane_id);
//                 printf("tSrS_float\n");
//                 print_tensor(tSrS_float(make_coord(_, 0), 0, 0));
//                 printf("====================\n");
//                 //print_tensor(sdS(1, _));
//             }

            //apply mask
            int row_offset = (warp_id % 2) * 16 + (lane_id / 4);
            int col_offset = (warp_id / 2) * 8 + (lane_id % 4) * 2;
            for (int i=0; i<2; i++) {
                for (int j=0;j<2;j++) {
                    for (int k=0;k<2;k++) {
                        for (int l=0;l<2;l++) {
                            int row = row_offset + 8 * j + 32 * k;
                            int col = col_offset + i + 32 * l;
                            if (row < col) {
                                tSrS_float(make_coord(i,j),k,l) = -FLT_MAX;
                                tdPrdP_float(make_coord(i,j),k,l) = 0;
                            }
                            // print("((%d, %d), %d, %d)\n", i, j, k, l);
                            // print("row = %d, col = %d\n", row_offset + 8 * j + 32 * k, col_offset + i + 32 * l );
                            // print("%d\n", 64 * (row_offset + 8 * j + 32 * k) + col_offset + i + 32 * l);
                            // printf("%d\n", tc(make_coord(i,j),k,l));
                            // print("====================\n");
                        }
                    }
                }
            }

//             if (blockIdx.z ==0 && warp_id == 0 && lane_id == 0 && kv_tile == 0) {
//                 printf("kv_tile = %d, warp_id = %d, lane_id = %d\n", kv_tile, warp_id, lane_id);
//                 printf("tSrS_float masked\n");
//                 print_tensor(tSrS_float(make_coord(_, 0), 0, 0));
//                 printf("====================\n");
//                 //print_tensor(sdS(1, _));
//             }


//             if (blockIdx.z ==0 && warp_id == 2 && lane_id == 0 && kv_tile == 0) {
//                 printf("kv_tile = %d, warp_id = %d, lane_id = %d\n", kv_tile, warp_id, lane_id);
//                 printf("tSrS_float masked\n");
//                 print_tensor(tSrS_float(make_coord(_, 0), 0, 0));
//                 printf("====================\n");
//                 //print_tensor(sdS(1, _));
//             }


            Tensor tSrP_float = tSrS_float;
            Tensor tdPrdS_float = tdPrdP_float;


            // compute P = exp(S-l)
            for (int i=0; i<2; i++) {
                for (int j=0;j<2;j++) {
                    for (int k=0; k< tSrS_float(make_coord(_,j),i,_).size(); k++) {
                        tSrP_float(make_coord(_,j),i,_)[k] = expf(tSrS_float(make_coord(_,j),i,_)[k] - rL[i][j]);
                    }
                }
            }


//             if (blockIdx.z ==0 && warp_id == 2 && lane_id == 0 && kv_tile == 0) {
//                 printf("kv_tile = %d, warp_id = %d, lane_id = %d\n", kv_tile, warp_id, lane_id);
//                 printf("tSrP_float\n");
//                 print_tensor(tSrP_float(make_coord(_, 0), 0, 0));
//                 printf("====================\n");
//                 //print_tensor(sdS(1, _));
//             }


            // compute dS = P \circ (dP - D)
            // tS has the same mma layout as tdP
            for (int i=0; i<2; i++) {
                for (int j=0;j<2;j++) {
                    for (int k=0; k< tdPrdP_float(make_coord(_,j),i,_).size(); k++) {
                        tdPrdS_float(make_coord(_,j),i,_)[k] = tSrP_float(make_coord(_,j),i,_)[k] * (tdPrdP_float(make_coord(_,j),i,_)[k] - rD[i][j]);
                    }
                }
            }

//             if (blockIdx.z ==0 && warp_id == 2 && lane_id == 0 && kv_tile == 0) {
//                 printf("kv_tile = %d, warp_id = %d, lane_id = %d\n", kv_tile, warp_id, lane_id);
//                 printf("tdPrdS_float\n");
//                 print_tensor(tdPrdS_float(make_coord(_, 0), 0, 0));
//                 printf("====================\n");
//                 //print_tensor(sdS(1, _));
//             }

            // convert dS from fp32 to fp16
//             constexpr int num_element_dS = decltype(size(tdPrdS_float))::value;
//
//             cutlass::NumericArrayConverter<half_t, float, num_element_dS> convert_op_dS;
//             auto frag_dS = convert_op_dS(*reinterpret_cast<const cutlass::Array<float, num_element_dS> *>(tdPrdS_float.data()));
//
//             Tensor tdPrdS = make_tensor(make_rmem_ptr<half_t>(&frag_dS), tdPrdS_float.layout());

            Tensor tdPrdS = convert_type<half_t>(tdPrdS_float);

            //copy(tSrP, tSsP);
            copy(tdPrdS, tdPsdS);

            __syncthreads();

//             if (blockIdx.z ==0 && warp_id == 0 && lane_id == 0 && kv_tile == 0) {
//                 printf("kv_tile = %d, warp_id = %d, lane_id = %d\n", kv_tile, warp_id, lane_id);
//                 print_tensor(sdS(make_coord(0,_)));
//                 //print_tensor(sdS(1, _));
//             }


            // dQ += dSK

            CUTE_UNROLL
            for (int dskt_block = 0; dskt_block < dSKt_BLOCK_MAX; dskt_block++) {
                copy(smem_tiled_copy_dS, tdQsdS_copy_view(_,_,dskt_block), tdQrdS_copy_view(_,_,dskt_block));
                copy(smem_tiled_copy_Kt, tdQsKt_copy_view(_,_,dskt_block), tdQrKt_copy_view(_,_,dskt_block));


                gemm(tiled_mma_dQ, tdQrdS(_,_,dskt_block), tdQrKt(_,_,dskt_block), tdQrdQ_float);
//                 if (blockIdx.z == 0 && warp_id==4 && lane_id==0) {
//                     print_tensor(tdQrdQ_float);
//                 }
            }

            __syncthreads();

        }
    }


    // rescale by head dim
    for (int i=0;i< tdQrdQ_float.size();i ++ ) {
        tdQrdQ_float[i] *= 1.0f / sqrtf(kHeadDim);
    }


//     if (blockIdx.z == 0 && warp_id==4 && lane_id==0) {
//         print_tensor(tdQrdQ_float);
//     }

    // dQ
//     constexpr int num_element = decltype(size(tdQrdQ_float))::value;
//
//     cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
//     auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tdQrdQ_float.data()));
//
//     Tensor tdQrdQ = make_tensor(make_rmem_ptr<half_t>(&frag), tdQrdQ_float.layout());

    Tensor tdQrdQ = convert_type<half_t>(tdQrdQ_float);

    copy(tdQrdQ, tdQsdQ);

    __syncthreads();

    copy(gmem_tiled_copy_QKV, tdQsdQ_copy, tdQgdQ_copy);

}



template <typename Kernel_traits, bool Is_causal>
inline __device__ void compute_dk_dv_1colblock(
    half_t * __restrict__ q_ptr,
    half_t * __restrict__ k_ptr,
    half_t * __restrict__ v_ptr,
    float * __restrict__ l_ptr,
    float * __restrict__ d_ptr,
    half_t * __restrict__ do_ptr,
    half_t* __restrict__ dk_ptr,
    half_t* __restrict__ dv_ptr,
    int batch_size, int seq_len, int num_heads, int head_dim, int is_causal,
    int bidb, int bidh, int n_block
)
{   
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    //constexpr int kNWarps = 8;
    //constexpr int kNThreads = kNWarps * 32;

    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;

    // for 8 warps, the 32x32 tiled mmas is like
    // -------------------------------------
    // | Warp 0 | Warp 2 | Warp 4 | Warp 6 |
    // -------------------------------------
    // | Warp 1 | Warp 3 | Warp 5 | Warp 7 |
    // -------------------------------------
    // for 64 x 64 tiledmma, each thread computes 16 numbers and each row can be accessed by
    // print(tc);
    // print_tensor(tc);
    // ptr[32b](0x7ea2408d8910) o ((_2,_2),_2,_2):((_1,_512),_2048,_32)
    // for (int i=0;i<2;i++) {
    //     for (int j=0;j<2;j++) {
    //         print_tensor(tc(make_coord(_,j),i,_));
    //     }
    // }

    // Q
    Tensor mQ = make_tensor(make_gmem_ptr(q_ptr),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gQ = local_tile(mQ(bidb, _, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(_, 0));


    // K
    Tensor mK = make_tensor(make_gmem_ptr(k_ptr),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gK = local_tile(mK(bidb, _, bidh, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(n_block, 0));

    // V
    Tensor mV = make_tensor(make_gmem_ptr(v_ptr),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gV = local_tile(mV(bidb, _, bidh, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(n_block, 0));


    // L = m + log l
    Tensor mL = make_tensor(make_gmem_ptr(l_ptr),
                             make_shape(batch_size, num_heads, seq_len),
                             make_stride(seq_len * num_heads,  seq_len, Int<1>{}));

    Tensor gL = local_tile(mL(bidb, bidh, _), Shape<Int<kBlockM>>{},
                           make_coord(_));


    Tensor mD = make_tensor(make_gmem_ptr(d_ptr),
                             make_shape(batch_size, num_heads, seq_len),
                             make_stride(seq_len * num_heads,  seq_len, Int<1>{}));

    Tensor gD = local_tile(mD(bidb, bidh, _), Shape<Int<kBlockM>>{},
                           make_coord(_));

    // dO
    Tensor mdO = make_tensor(make_gmem_ptr(do_ptr),
                             make_shape(batch_size, seq_len, num_heads, head_dim),
                             make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gdO = local_tile(mdO(bidb, _, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(_, 0));

    // dV
    Tensor mdV = make_tensor(make_gmem_ptr(dv_ptr),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gdV = local_tile(mdV(bidb, _, bidh, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(n_block, 0));

    // dK
    Tensor mdK = make_tensor(make_gmem_ptr(dk_ptr),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gdK = local_tile(mdK(bidb, _, bidh, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(n_block, 0));



    extern __shared__ char smem_[];


    // 64 * 128 = 16KB
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), typename Kernel_traits::SmemLayoutQ{});
    Tensor sQt = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), typename Kernel_traits::SmemLayoutQTransposed{});
    //Tensor sK = make_tensor(sQ.data() + kBlockM * kHeadDim, SmemLayoutKV{});

    // 64 * 128 = 16KB
    Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKVTransposed{});

    // 64 * 128 = 16KB
    Tensor sdO = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutQ{});
    Tensor sdOt = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutQTransposed{});
//
//
//     // 64 * 128 = 16KB
    Tensor sV = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});
//
//
    // 64 * 64 = 8KB
    Tensor sP = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sPt = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutPdSTransposed{});
//
    // 64 * 64 = 8KB
    Tensor sdS = make_tensor(sP.data() + size(sP), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sdSt = make_tensor(sP.data() + size(sP), typename Kernel_traits::SmemLayoutPdSTransposed{});



    Tensor sdK = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), typename Kernel_traits::SmemLayoutKV{});
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutKV{});




    //int thread_id = threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    //int thread_row = warp_id * 16 + lane_id / 4;
    int warp_offset = (warp_id % 2) * 16;
    int thread_offset = lane_id / 4;

    float rL[2][2] = {0};
    float rD[2][2] = {0};

    // Copy operation
    typename Kernel_traits::GmemTiledCopy gmem_tiled_copy_QKV;
    //GmemTiledCopyQKV gmem_tiled_copy_QKV;

    ThrCopy thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(threadIdx.x);

    Tensor tQgQ = thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = thr_copy_QKV.partition_D(sQ);
    Tensor tQrQ = make_fragment_like(tQsQ);

    Tensor tKgK = thr_copy_QKV.partition_S(gK);
    Tensor tKsK = thr_copy_QKV.partition_D(sK);

    Tensor tVgV = thr_copy_QKV.partition_S(gV);
    Tensor tVsV = thr_copy_QKV.partition_D(sV);
    Tensor tVrV = make_fragment_like(tVsV);

    Tensor tdOgdO = thr_copy_QKV.partition_S(gdO);
    Tensor tdOsdO = thr_copy_QKV.partition_D(sdO);
    //Tensor tdOrdO = make_fragment_like(tdOsdO);

    Tensor tdKsdK_copy = thr_copy_QKV.partition_S(sdK);
    Tensor tdKgdK_copy = thr_copy_QKV.partition_D(gdK);

    Tensor tdVsdV_copy = thr_copy_QKV.partition_S(sdV);
    Tensor tdVgdV_copy = thr_copy_QKV.partition_D(gdV);



    // S = QK^T
    typename Kernel_traits::TiledMma_SdP tiled_mma_S;
    ThrMMA thr_mma_S = tiled_mma_S.get_slice(threadIdx.x);
    Tensor tSgQ = thr_mma_S.partition_A(gQ);
    Tensor tSsQ = thr_mma_S.partition_A(sQ);
    Tensor tSrQ = thr_mma_S.make_fragment_A(tSsQ);

    Tensor tSgK = thr_mma_S.partition_B(gK);
    Tensor tSsK = thr_mma_S.partition_B(sK);
    Tensor tSrK = thr_mma_S.make_fragment_B(tSsK);

    Tensor tSrS_float = partition_fragment_C(tiled_mma_S, Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tSsP = thr_mma_S.partition_C(sP);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQdOPdS{}, tiled_mma_S);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(threadIdx.x);
    auto tSsQ_copy_view = smem_thr_copy_Q.partition_S(sQ);
    auto tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomKV{}, tiled_mma_S);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_slice(threadIdx.x);
    auto tSsK_copy_view = smem_thr_copy_K.partition_S(sK);
    auto tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);


    // dP = dOV^T
    typename Kernel_traits::TiledMma_SdP tiled_mma_dP;
    ThrMMA thr_mma_dP = tiled_mma_dP.get_slice(threadIdx.x);

    Tensor tdPgdO = thr_mma_dP.partition_A(gdO);
    Tensor tdPsdO = thr_mma_dP.partition_A(sdO);
    Tensor tdPrdO = thr_mma_dP.make_fragment_A(tdPsdO);

    Tensor tdPgV = thr_mma_dP.partition_B(gV);
    Tensor tdPsV = thr_mma_dP.partition_B(sV);
    Tensor tdPrV = thr_mma_dP.make_fragment_B(tdPsV);

    Tensor tdPrdP_float = partition_fragment_C(tiled_mma_dP, Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tdPsdS = thr_mma_dP.partition_C(sdS);

    auto smem_tiled_copy_dO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQdOPdS{}, tiled_mma_dP);
    auto smem_thr_copy_dO = smem_tiled_copy_dO.get_slice(threadIdx.x);
    auto tdPsdO_copy_view = smem_thr_copy_dO.partition_S(sdO);
    auto tdPrdO_copy_view = smem_thr_copy_dO.retile_D(tdPrdO);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomKV{}, tiled_mma_dP);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_slice(threadIdx.x);
    auto tdPsV_copy_view = smem_thr_copy_V.partition_S(sV);
    auto tdPrV_copy_view = smem_thr_copy_V.retile_D(tdPrV);



    // dV += P^TdO
    typename Kernel_traits::TiledMma_dKdV tiled_mma_dV;
    ThrMMA thr_mma_dV = tiled_mma_dV.get_slice(threadIdx.x);

    Tensor tdVsPt = thr_mma_dV.partition_A(sPt);
    Tensor tdVrPt = thr_mma_dV.make_fragment_A(tdVsPt);

    Tensor tdVsdOt = thr_mma_dV.partition_B(sdOt);
    Tensor tdVrdOt = thr_mma_dV.make_fragment_B(tdVsdOt);

    Tensor tdVrdV_float = partition_fragment_C(tiled_mma_dV, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor tdVsdV = thr_mma_dV.partition_C(sdV);

    auto smem_tiled_copy_Pt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomPtdSt{}, tiled_mma_dV);
    auto smem_thr_copy_Pt = smem_tiled_copy_Pt.get_slice(threadIdx.x);
    auto tdVsPt_copy_view = smem_thr_copy_Pt.partition_S(sPt);
    auto tdVrPt_copy_view = smem_thr_copy_Pt.retile_D(tdVrPt);

    auto smem_tiled_copy_dOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomdOt{}, tiled_mma_dV);
    auto smem_thr_copy_dOt = smem_tiled_copy_dOt.get_slice(threadIdx.x);
    auto tdVsdOt_copy_view = smem_thr_copy_dOt.partition_S(sdOt);
    auto tdVrdOt_copy_view = smem_thr_copy_dOt.retile_D(tdVrdOt);


    // dK += dS^TQ
    typename Kernel_traits::TiledMma_dKdV tiled_mma_dK;
    ThrMMA thr_mma_dK = tiled_mma_dK.get_slice(threadIdx.x);
    Tensor tdKsdSt = thr_mma_dK.partition_A(sdSt);
    Tensor tdKrdSt = thr_mma_dK.make_fragment_A(tdKsdSt);

    Tensor tdKsQt = thr_mma_dK.partition_B(sQt);
    Tensor tdKrQt = thr_mma_dV.make_fragment_B(tdKsQt);

    Tensor tdKrdK_float = partition_fragment_C(tiled_mma_dK, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor tdKsdK = thr_mma_dK.partition_C(sdK);

    auto smem_tiled_copy_dSt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomPtdSt{}, tiled_mma_dK);
    auto smem_thr_copy_dSt = smem_tiled_copy_dSt.get_slice(threadIdx.x);
    auto tdKsdSt_copy_view = smem_thr_copy_dSt.partition_S(sdSt);
    auto tdKrdSt_copy_view = smem_thr_copy_dSt.retile_D(tdKrdSt);

    auto smem_tiled_copy_Qt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomQt{}, tiled_mma_dK);
    auto smem_thr_copy_Qt = smem_tiled_copy_Qt.get_slice(threadIdx.x);
    auto tdKsQt_copy_view = smem_thr_copy_dOt.partition_S(sQt);
    auto tdKrQt_copy_view = smem_thr_copy_dOt.retile_D(tdKrQt);



    int Q_TILE_NO_MASK = 0;
    int Q_TILE_MASK_START = 0;
    int Q_TILE_MASK_END = 0;

    if constexpr (Is_causal) {
        // number of KV_TILES that does not need a mask
        Q_TILE_NO_MASK = n_block + 1;
        Q_TILE_MASK_START = n_block;
        Q_TILE_MASK_END = Q_TILE_NO_MASK;

    } else {
        Q_TILE_NO_MASK = 0;
    }


    auto Q_TILE_MAX = size<3>(tSgQ);
    auto QK_BLOCK_MAX = size<2>(tSsK);
    auto PtdOt_BLOCK_MAX = size<2>(tdVsPt);
    // load K, V, dK, dV tiles

    copy(gmem_tiled_copy_QKV, tKgK, tKsK);
    //copy(tSgK, tSsK);
    copy(gmem_tiled_copy_QKV, tVgV, tVrV);


    clear(tdVrdV_float);
    clear(tdKrdK_float);


    copy(gmem_tiled_copy_QKV, tQgQ(_,_,_,Q_TILE_NO_MASK), tQrQ);
    //copy(gmem_tiled_copy_QKV, tdOgdO(_,_,_,0), tdOrdO);

    CUTE_NO_UNROLL
    for (int q_tile = Q_TILE_NO_MASK; q_tile < Q_TILE_MAX; ++q_tile) {
        copy(gmem_tiled_copy_QKV, tVrV, tVsV);

        clear(tSrS_float);
        clear(tdPrdP_float);

        copy(gmem_tiled_copy_QKV, tQrQ, tQsQ);
        //copy(gmem_tiled_copy_QKV, tdOrdO, tdOsdO);
        // load gQ to sQ
        //copy(gmem_tiled_copy_QKV, tQgQ(_,_,_,q_tile), tQsQ);
        copy(gmem_tiled_copy_QKV, tdOgdO(_,_,_,q_tile), tdOsdO);


        __syncthreads();

        // somehow pipelining gmem loads for both Q and dO use alot more registers which is slower
        if (q_tile + 1 < Q_TILE_MAX) {
            copy(gmem_tiled_copy_QKV, tQgQ(_,_,_,q_tile+1), tQrQ);
            //copy(gmem_tiled_copy_QKV, tdOgdO(_,_,_,q_tile+1), tdOrdO);
        }

        // compute S=QK^T


//         copy(smem_tiled_copy_Q, tSsQ_copy_view(_,_,0), tSrQ_copy_view(_,_,0));
//         copy(smem_tiled_copy_K, tSsK_copy_view(_,_,0), tSrK_copy_view(_,_,0));

        CUTE_UNROLL
        for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
            copy(smem_tiled_copy_Q, tSsQ_copy_view(_,_,qk_block), tSrQ_copy_view(_,_,qk_block));
            copy(smem_tiled_copy_K, tSsK_copy_view(_,_,qk_block), tSrK_copy_view(_,_,qk_block));

//             int qk_block_next = (qk_block + 1) % QK_BLOCK_MAX;
//             copy(smem_tiled_copy_Q, tSsQ_copy_view(_,_,qk_block_next), tSrQ_copy_view(_,_,qk_block_next));
            //copy(smem_tiled_copy_K, tSsK_copy_view(_,_,qk_block_next), tSrK_copy_view(_,_,qk_block_next));

            copy(smem_tiled_copy_dO, tdPsdO_copy_view(_,_,qk_block), tdPrdO_copy_view(_,_,qk_block));
            copy(smem_tiled_copy_V, tdPsV_copy_view(_,_,qk_block), tdPrV_copy_view(_,_,qk_block));

            gemm(tiled_mma_S, tSrQ(_,_,qk_block), tSrK(_,_,qk_block), tSrS_float);
            gemm(tiled_mma_dP, tdPrdO(_,_,qk_block), tdPrV(_,_,qk_block), tdPrdP_float);
        }


        //if (blockIdx.z == 0 && warp_id == 7 && lane_id == 31) {



//         gemm(tiled_mma_S, tSsQ, tSsK, tSrS_float);
//         gemm(tiled_mma_dP, tdPsdO, tdPsV, tdPrdP_float);


        __syncthreads();

        // load rL, rD from gmem to rmem
        for (int i=0;i<2;i++) {
            for (int j=0;j<2;j++) {
                rL[i][j] = gL((warp_offset + thread_offset + 8 * j + 32 * i), q_tile);
                rD[i][j] = gD((warp_offset + thread_offset + 8 * j + 32 * i), q_tile);
            }
        }

        // rescale S
        for (int i=0;i< tSrS_float.size();i ++ ) {
            tSrS_float[i] *= 1.0f / sqrtf(kHeadDim);
        }


        Tensor tSrP_float = tSrS_float;
        Tensor tdPrdS_float = tdPrdP_float;

        // compute P = exp(S-l)
        for (int i=0; i<2; i++) {
            for (int j=0;j<2;j++) {
                for (int k=0; k< tSrS_float(make_coord(_,j),i,_).size(); k++) {
                    tSrP_float(make_coord(_,j),i,_)[k] = expf(tSrS_float(make_coord(_,j),i,_)[k] - rL[i][j]);
                }
            }
        }



        // compute dS = P \circ (dP - D)
        // tS has the same mma layout as tdP
        for (int i=0; i<2; i++) {
            for (int j=0;j<2;j++) {
                for (int k=0; k< tdPrdP_float(make_coord(_,j),i,_).size(); k++) {
                    tdPrdS_float(make_coord(_,j),i,_)[k] = tSrP_float(make_coord(_,j),i,_)[k] * (tdPrdP_float(make_coord(_,j),i,_)[k] - rD[i][j]);
                }
            }
        }

        //convert P from fp32 to fp16
//         constexpr int num_element = decltype(size(tSrP_float))::value;
//
//         cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
//         auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tSrP_float.data()));
//
//         Tensor tSrP = make_tensor(make_rmem_ptr<half_t>(&frag), tSrP_float.layout());

        Tensor tSrP = convert_type<half_t>(tSrP_float);
//
//
        // convert dS from fp32 to fp16
//         constexpr int num_element_dS = decltype(size(tdPrdS_float))::value;
//
//         cutlass::NumericArrayConverter<half_t, float, num_element_dS> convert_op_dS;
//         auto frag_dS = convert_op_dS(*reinterpret_cast<const cutlass::Array<float, num_element_dS> *>(tdPrdS_float.data()));
//
//         Tensor tdPrdS = make_tensor(make_rmem_ptr<half_t>(&frag_dS), tdPrdS_float.layout());

        Tensor tdPrdS = convert_type<half_t>(tdPrdS_float);

//
//
        copy(gmem_tiled_copy_QKV, tVsV, tVrV);
        __syncthreads();

//        if (thread(0)) {
//             //printf("q_tile = %d, warp_id = %d, lane_id = %d\n", q_tile, warp_id, lane_id);
//             //printf("tSrS_float\n");
//             print_tensor(tSrP);
//             //printf("====================\n");
//             //print_tensor(sdS(1, _));
//         }

        copy(tSrP, tSsP);
        copy(tdPrdS, tdPsdS);
//
//
        __syncthreads();

//         if (thread(0) && q_tile == 0 ) {
//             printf("tdVsPt\n");
//             print_tensor(tdVsPt);
//             printf("tdVrPt\n");
//             print_tensor(tdVrPt);
//             printf("tdVsdOt\n");
//             print_tensor(tdVsdOt);
//             printf("tdVrdOt\n");
//             print_tensor(tdVrdOt);
//
//             printf("tdVsPt_copy_view\n");
//             print_tensor(tdVsPt_copy_view);
//             printf("tdVrPt_copy_view\n");
//             print_tensor(tdVrPt_copy_view);
//             printf("tdVsdOt_copy_view\n");
//             print_tensor(tdVsdOt_copy_view);
//             printf("tdVrdOt_copy_view\n");
//             print_tensor(tdVrdOt_copy_view);
//         }



        // dV += P^TdO
        CUTE_UNROLL
        for (int ptdot_block = 0; ptdot_block < PtdOt_BLOCK_MAX; ptdot_block++) {
            copy(smem_tiled_copy_dSt, tdVsPt_copy_view(_,_,ptdot_block), tdVrPt_copy_view(_,_,ptdot_block));
            copy(smem_tiled_copy_Qt, tdVsdOt_copy_view(_,_,ptdot_block), tdVrdOt_copy_view(_,_,ptdot_block));

            //copy(tdVsPt(_,_,ptdot_block), tdVrPt(_,_,ptdot_block));
            //copy(tdVsdOt(_,_,ptdot_block), tdVrdOt(_,_,ptdot_block));
            gemm(tiled_mma_dV, tdVrPt(_,_,ptdot_block), tdVrdOt(_,_,ptdot_block), tdVrdV_float);

        }


        //gemm(tiled_mma_dV, tdVsPt, tdVsdOt, tdVrdV_float);
//



        // dK += dS^TQ

        CUTE_UNROLL
        for (int ptdot_block = 0; ptdot_block < PtdOt_BLOCK_MAX; ptdot_block++) {
            copy(smem_tiled_copy_dSt, tdKsdSt_copy_view(_,_,ptdot_block), tdKrdSt_copy_view(_,_,ptdot_block));
            copy(smem_tiled_copy_Qt, tdKsQt_copy_view(_,_,ptdot_block), tdKrQt_copy_view(_,_,ptdot_block));

            //copy(tdVsPt(_,_,ptdot_block), tdVrPt(_,_,ptdot_block));
            //copy(tdVsdOt(_,_,ptdot_block), tdVrdOt(_,_,ptdot_block));
            gemm(tiled_mma_dK, tdKrdSt(_,_,ptdot_block), tdKrQt(_,_,ptdot_block), tdKrdK_float);

        }

        //gemm(tiled_mma_dK, tdKsdSt, tdKsQt, tdKrdK_float);


        __syncthreads();

    }

//     To compute the mask, imagine we are working in a 64 x 64 row-major matrix.
//     We want to compute a map
//     (warp_id, lane_id) -> offset in a 64 x 64 row major matrix -> 2d coordinates
//     To derive the first map we can partition a 64 x 64 row-major matrix using the bwd mma
//     and print out the layout of each thread partition.
//     For hdim = 128, the layout is ((_2,_2),_2,_2):((_1,_512),_2048,_32)
//     The starting offset of each thread is given by
//     (warp_id / 4) * 64 * 16 + (warp_id % 4) * 8 + (lane_id / 4) * 64 + (lane_id % 4) * 2
//     the offset is simply offset_start + dot product between layout and stride
//     The second map is simply offset -> (offset / 64, offset % 64)


    if (Is_causal) {
        copy(gmem_tiled_copy_QKV, tQgQ(_,_,_,Q_TILE_MASK_START), tQrQ);
        //copy(gmem_tiled_copy_QKV, tdOgdO(_,_,_,0), tdOrdO);

        CUTE_NO_UNROLL
        for (int q_tile = Q_TILE_MASK_START; q_tile < Q_TILE_MASK_END; ++q_tile) {
            copy(gmem_tiled_copy_QKV, tVrV, tVsV);

            clear(tSrS_float);
            clear(tdPrdP_float);

            copy(gmem_tiled_copy_QKV, tQrQ, tQsQ);
            //copy(gmem_tiled_copy_QKV, tdOrdO, tdOsdO);
            // load gQ to sQ
            //copy(gmem_tiled_copy_QKV, tQgQ(_,_,_,q_tile), tQsQ);
            copy(gmem_tiled_copy_QKV, tdOgdO(_,_,_,q_tile), tdOsdO);


            __syncthreads();

            // somehow pipelining gmem loads for both Q and dO use alot more registers which is slower
            if (q_tile + 1 < Q_TILE_MASK_END) {
                copy(gmem_tiled_copy_QKV, tQgQ(_,_,_,q_tile+1), tQrQ);
                //copy(gmem_tiled_copy_QKV, tdOgdO(_,_,_,q_tile+1), tdOrdO);
            }

            // compute S=QK^T


    //         copy(smem_tiled_copy_Q, tSsQ_copy_view(_,_,0), tSrQ_copy_view(_,_,0));
    //         copy(smem_tiled_copy_K, tSsK_copy_view(_,_,0), tSrK_copy_view(_,_,0));

            CUTE_UNROLL
            for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
                copy(smem_tiled_copy_Q, tSsQ_copy_view(_,_,qk_block), tSrQ_copy_view(_,_,qk_block));
                copy(smem_tiled_copy_K, tSsK_copy_view(_,_,qk_block), tSrK_copy_view(_,_,qk_block));

    //             int qk_block_next = (qk_block + 1) % QK_BLOCK_MAX;
    //             copy(smem_tiled_copy_Q, tSsQ_copy_view(_,_,qk_block_next), tSrQ_copy_view(_,_,qk_block_next));
                //copy(smem_tiled_copy_K, tSsK_copy_view(_,_,qk_block_next), tSrK_copy_view(_,_,qk_block_next));

                copy(smem_tiled_copy_dO, tdPsdO_copy_view(_,_,qk_block), tdPrdO_copy_view(_,_,qk_block));
                copy(smem_tiled_copy_V, tdPsV_copy_view(_,_,qk_block), tdPrV_copy_view(_,_,qk_block));

                gemm(tiled_mma_S, tSrQ(_,_,qk_block), tSrK(_,_,qk_block), tSrS_float);
                gemm(tiled_mma_dP, tdPrdO(_,_,qk_block), tdPrV(_,_,qk_block), tdPrdP_float);
            }





    //         gemm(tiled_mma_S, tSsQ, tSsK, tSrS_float);
    //         gemm(tiled_mma_dP, tdPsdO, tdPsV, tdPrdP_float);


            __syncthreads();

            // load rL, rD from gmem to rmem
            for (int i=0;i<2;i++) {
                for (int j=0;j<2;j++) {
                    rL[i][j] = gL((warp_offset + thread_offset + 8 * j + 32 * i), q_tile);
                    rD[i][j] = gD((warp_offset + thread_offset + 8 * j + 32 * i), q_tile);
                }
            }

            // rescale S
            for (int i=0;i< tSrS_float.size();i ++ ) {
                tSrS_float[i] *= 1.0f / sqrtf(kHeadDim);
            }


            //apply mask
            int row_offset = (warp_id % 2) * 16 + (lane_id / 4);
            int col_offset = (warp_id / 2) * 8 + (lane_id % 4) * 2;
            for (int i=0; i<2; i++) {
                for (int j=0;j<2;j++) {
                    for (int k=0;k<2;k++) {
                        for (int l=0;l<2;l++) {
                            int row = row_offset + 8 * j + 32 * k;
                            int col = col_offset + i + 32 * l;
                            if (row < col) {
                                tSrS_float(make_coord(i,j),k,l) = -FLT_MAX;
                                tdPrdP_float(make_coord(i,j),k,l) = 0;
                            }
                            // print("((%d, %d), %d, %d)\n", i, j, k, l);
                            // print("row = %d, col = %d\n", row_offset + 8 * j + 32 * k, col_offset + i + 32 * l );
                            // print("%d\n", 64 * (row_offset + 8 * j + 32 * k) + col_offset + i + 32 * l);
                            // printf("%d\n", tc(make_coord(i,j),k,l));
                            // print("====================\n");
                        }
                    }
                }
            }

//             if (blockIdx.z == 0 && warp_id == 6 && lane_id == 4) {
//                 printf("kv_tile = %d, warp_id = %d, lane_id = %d\n", q_tile, warp_id, lane_id);
//                 printf("tSrS_float\n");
//                 print_tensor(tSrS_float(make_coord(_, _), 0, 1));
//                 printf("====================\n");
//                 //print_tensor(sdS(1, _));
//             }


            Tensor tSrP_float = tSrS_float;
            Tensor tdPrdS_float = tdPrdP_float;

            // compute P = exp(S-l)
            for (int i=0; i<2; i++) {
                for (int j=0;j<2;j++) {
                    for (int k=0; k< tSrS_float(make_coord(_,j),i,_).size(); k++) {
                        tSrP_float(make_coord(_,j),i,_)[k] = expf(tSrS_float(make_coord(_,j),i,_)[k] - rL[i][j]);
                    }
                }
            }
    //
            // compute dS = P \circ (dP - D)
            // tS has the same mma layout as tdP
            for (int i=0; i<2; i++) {
                for (int j=0;j<2;j++) {
                    for (int k=0; k< tdPrdP_float(make_coord(_,j),i,_).size(); k++) {
                        tdPrdS_float(make_coord(_,j),i,_)[k] = tSrP_float(make_coord(_,j),i,_)[k] * (tdPrdP_float(make_coord(_,j),i,_)[k] - rD[i][j]);
                    }
                }
            }

            //convert P from fp32 to fp16
//             constexpr int num_element = decltype(size(tSrP_float))::value;
//
//             cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
//             auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tSrP_float.data()));
//
//             Tensor tSrP = make_tensor(make_rmem_ptr<half_t>(&frag), tSrP_float.layout());
            Tensor tSrP = convert_type<half_t>(tSrP_float);
    //
    //
            // convert dS from fp32 to fp16
//             constexpr int num_element_dS = decltype(size(tdPrdS_float))::value;
//
//             cutlass::NumericArrayConverter<half_t, float, num_element_dS> convert_op_dS;
//             auto frag_dS = convert_op_dS(*reinterpret_cast<const cutlass::Array<float, num_element_dS> *>(tdPrdS_float.data()));
//
//             Tensor tdPrdS = make_tensor(make_rmem_ptr<half_t>(&frag_dS), tdPrdS_float.layout());

            Tensor tdPrdS = convert_type<half_t>(tdPrdS_float);

    //
    //
            copy(gmem_tiled_copy_QKV, tVsV, tVrV);
            __syncthreads();

            copy(tSrP, tSsP);
            copy(tdPrdS, tdPsdS);
    //
    //
            __syncthreads();


//             if (thread(0)) {
//                 printf("q_tile = %d\n", q_tile);
//                 print_tensor(sPt(62, _));
//                 printf("tdVsPt\n");
//                 print_tensor(tdVsPt);
//                 printf("tdVrPt\n");
//                 print_tensor(tdVrPt);
//                 printf("tdVsdOt\n");
//                 print_tensor(tdVsdOt);
//                 printf("tdVrdOt\n");
//                 print_tensor(tdVrdOt);
//
//                 printf("tdVsPt_copy_view\n");
//                 print_tensor(tdVsPt_copy_view);
//                 printf("tdVrPt_copy_view\n");
//                 print_tensor(tdVrPt_copy_view);
//                 printf("tdVsdOt_copy_view\n");
//                 print_tensor(tdVsdOt_copy_view);
//                 printf("tdVrdOt_copy_view\n");
//                 print_tensor(tdVrdOt_copy_view);
//             }



            // dV += P^TdO
            CUTE_UNROLL
            for (int ptdot_block = 0; ptdot_block < PtdOt_BLOCK_MAX; ptdot_block++) {
                copy(smem_tiled_copy_dSt, tdVsPt_copy_view(_,_,ptdot_block), tdVrPt_copy_view(_,_,ptdot_block));
                copy(smem_tiled_copy_Qt, tdVsdOt_copy_view(_,_,ptdot_block), tdVrdOt_copy_view(_,_,ptdot_block));

                //copy(tdVsPt(_,_,ptdot_block), tdVrPt(_,_,ptdot_block));
                //copy(tdVsdOt(_,_,ptdot_block), tdVrdOt(_,_,ptdot_block));
                gemm(tiled_mma_dV, tdVrPt(_,_,ptdot_block), tdVrdOt(_,_,ptdot_block), tdVrdV_float);

            }


            //gemm(tiled_mma_dV, tdVsPt, tdVsdOt, tdVrdV_float);
    //



            // dK += dS^TQ

            CUTE_UNROLL
            for (int ptdot_block = 0; ptdot_block < PtdOt_BLOCK_MAX; ptdot_block++) {
                copy(smem_tiled_copy_dSt, tdKsdSt_copy_view(_,_,ptdot_block), tdKrdSt_copy_view(_,_,ptdot_block));
                copy(smem_tiled_copy_Qt, tdKsQt_copy_view(_,_,ptdot_block), tdKrQt_copy_view(_,_,ptdot_block));

                //copy(tdVsPt(_,_,ptdot_block), tdVrPt(_,_,ptdot_block));
                //copy(tdVsdOt(_,_,ptdot_block), tdVrdOt(_,_,ptdot_block));
                gemm(tiled_mma_dK, tdKrdSt(_,_,ptdot_block), tdKrQt(_,_,ptdot_block), tdKrdK_float);

            }

            //gemm(tiled_mma_dK, tdKsdSt, tdKsQt, tdKrdK_float);


            __syncthreads();

        }
    }


    // dV
//     constexpr int num_element = decltype(size(tdVrdV_float))::value;
//
//     cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
//     auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tdVrdV_float.data()));
//
//     Tensor tdVrdV = make_tensor(make_rmem_ptr<half_t>(&frag), tdVrdV_float.layout());

    Tensor tdVrdV = convert_type<half_t>(tdVrdV_float);

    copy(tdVrdV, tdVsdV);
//
//
    // dK

    // rescale by head dim
    for (int i=0;i< tdKrdK_float.size();i ++ ) {
        tdKrdK_float[i] *= 1.0f / sqrtf(kHeadDim);
    }


//     constexpr int num_element_dK = decltype(size(tdKrdK_float))::value;
//
//     cutlass::NumericArrayConverter<half_t, float, num_element_dK> convert_op_dK;
//     auto frag_dK = convert_op_dK(*reinterpret_cast<const cutlass::Array<float, num_element_dK> *>(tdKrdK_float.data()));
//
//     Tensor tdKrdK = make_tensor(make_rmem_ptr<half_t>(&frag_dK), tdKrdK_float.layout());

    Tensor tdKrdK = convert_type<half_t>(tdKrdK_float);

    copy(tdKrdK, tdKsdK);

    __syncthreads();

//     copy(tdQrdQ, tdQsdQ);
//
//     __syncthreads();
//
//     copy(gmem_tiled_copy_QKV, tdQsdQ_copy, tdQgdQ_copy);
    copy(gmem_tiled_copy_QKV, tdKsdK_copy, tdKgdK_copy);
    copy(gmem_tiled_copy_QKV, tdVsdV_copy, tdVgdV_copy);

}


//template <typename Kernel_traits, bool Is_causal>
//__global__ __launch_bounds__(256)
//void compute_dq_dk_dv_kernel(
//    half_t * __restrict__ q_ptr,
//    half_t * __restrict__ k_ptr,
//    half_t * __restrict__ v_ptr,
//    float __restrict__ l_ptr,
//    float __restrict__ d_ptr,
//    half_t * __restrict__ do_ptr,
//    float* __restrict__ dq_float_ptr,
//    half_t* __restrict__ dq_ptr,
//    half_t* __restrict__ dk_ptr,
//    half_t* __restrict__ dv_ptr,
//    int batch_size, int seq_len, int num_heads, int head_dim, int is_causal
//)
//{
//    constexpr int kBlockM = Kernel_traits::kBlockM;
//    constexpr int kBlockN = Kernel_traits::kBlockN;
//    constexpr int kHeadDim = Kernel_traits::kHeadDim;
//    //constexpr int kNWarps = 8;
//    //constexpr int kNThreads = kNWarps * 32;
//
//    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
//
//    // for 8 warps, the 32x32 tiled mmas is like
//    // -------------------------------------
//    // | Warp 0 | Warp 2 | Warp 4 | Warp 6 |
//    // -------------------------------------
//    // | Warp 1 | Warp 3 | Warp 5 | Warp 7 |
//    // -------------------------------------
//    // for 64 x 64 tiledmma, each thread computes 16 numbers and each row can be accessed by
//    // print(tc);
//    // print_tensor(tc);
//    // ptr[32b](0x7ea2408d8910) o ((_2,_2),_2,_2):((_1,_512),_2048,_32)
//    // for (int i=0;i<2;i++) {
//    //     for (int j=0;j<2;j++) {
//    //         print_tensor(tc(make_coord(_,j),i,_));
//    //     }
//    // }
//
//    // Q
//    Tensor mQ = make_tensor(make_gmem_ptr(q_ptr),
//                            make_shape(batch_size, seq_len, num_heads, head_dim),
//                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));
//
//    Tensor gQ = local_tile(mQ(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                           make_coord(_, 0));
//
//
//    // K
//    Tensor mK = make_tensor(make_gmem_ptr(k_ptr),
//                            make_shape(batch_size, seq_len, num_heads, head_dim),
//                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));
//
//    Tensor gK = local_tile(mK(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                           make_coord(blockIdx.z, 0));
//
//    // V
//    Tensor mV = make_tensor(make_gmem_ptr(v_ptr),
//                            make_shape(batch_size, seq_len, num_heads, head_dim),
//                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));
//
//    Tensor gV = local_tile(mV(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                           make_coord(blockIdx.z, 0));
//
//
//    // L = m + log l
//    Tensor mL = make_tensor(make_gmem_ptr(l_ptr),
//                             make_shape(batch_size, num_heads, seq_len),
//                             make_stride(seq_len * num_heads,  seq_len, Int<1>{}));
//
//    Tensor gL = local_tile(mL(blockIdx.x, blockIdx.y, _), Shape<Int<kBlockM>>{},
//                           make_coord(_));
//
//
//    Tensor mD = make_tensor(make_gmem_ptr(d_ptr),
//                             make_shape(batch_size, num_heads, seq_len),
//                             make_stride(seq_len * num_heads,  seq_len, Int<1>{}));
//
//    Tensor gD = local_tile(mD(blockIdx.x, blockIdx.y, _), Shape<Int<kBlockM>>{},
//                           make_coord(_));
//
//    // dO
//    Tensor mdO = make_tensor(make_gmem_ptr(do_ptr),
//                             make_shape(batch_size, seq_len, num_heads, head_dim),
//                             make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));
//
//    Tensor gdO = local_tile(mdO(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                           make_coord(_, 0));
//
//    // dV
//    Tensor mdV = make_tensor(make_gmem_ptr(dv_ptr),
//                            make_shape(batch_size, seq_len, num_heads, head_dim),
//                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));
//
//    Tensor gdV = local_tile(mdV(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                           make_coord(blockIdx.z, 0));
//
//    // dK
//    Tensor mdK = make_tensor(make_gmem_ptr(dk_ptr),
//                            make_shape(batch_size, seq_len, num_heads, head_dim),
//                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));
//
//    Tensor gdK = local_tile(mdK(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                           make_coord(blockIdx.z, 0));
//
//
//    // dQ
//    Tensor mdQ_float = make_tensor(make_gmem_ptr(dq_float_ptr),
//                            make_shape(batch_size, seq_len, num_heads, head_dim),
//                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));
//
//    Tensor gdQ_float = local_tile(mdQ_float(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                           make_coord(_, 0));
//
//
//
//    extern __shared__ char smem_[];
//
//
//    // 64 * 128 = 16KB
//    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), typename Kernel_traits::SmemLayoutQ{});
//    Tensor sQt = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), typename Kernel_traits::SmemLayoutQTransposed{});
//    //Tensor sK = make_tensor(sQ.data() + kBlockM * kHeadDim, SmemLayoutKV{});
//
//    // 64 * 128 = 16KB
//    Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
//    Tensor sKt = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKVTransposed{});
//
//    // 64 * 128 = 16KB
//    Tensor sdO = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutQ{});
//    Tensor sdOt = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutQTransposed{});
////
////
////     // 64 * 128 = 16KB
//    Tensor sV = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});
////
////
//    // 64 * 64 = 8KB
//    Tensor sP = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutPdS{});
//    Tensor sPt = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutPdSTransposed{});
////
//    // 64 * 64 = 8KB
//    Tensor sdS = make_tensor(sP.data() + size(sP), typename Kernel_traits::SmemLayoutPdS{});
//    Tensor sdSt = make_tensor(sP.data() + size(sP), typename Kernel_traits::SmemLayoutPdSTransposed{});
//
//
//    Tensor sdQ = make_tensor(sdS.data() + size(sdS), typename Kernel_traits::SmemLayoutQ{});
//
//
//    Tensor sdK = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), typename Kernel_traits::SmemLayoutKV{});
//    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutKV{});
//
//
//
//
//
//    //int thread_id = threadIdx.x;
//    int lane_id = threadIdx.x % 32;
//    int warp_id = threadIdx.x / 32;
//
//    //int thread_row = warp_id * 16 + lane_id / 4;
//    int warp_offset = (warp_id % 2) * 16;
//    int thread_offset = lane_id / 4;
//
//    float rL[2][2] = {0};
//    float rD[2][2] = {0};
//
//    // Copy operation
//    typename Kernel_traits::GmemTiledCopy gmem_tiled_copy_QKV;
//    //GmemTiledCopyQKV gmem_tiled_copy_QKV;
//
//    ThrCopy thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(threadIdx.x);
//
//    Tensor tQgQ = thr_copy_QKV.partition_S(gQ);
//    Tensor tQsQ = thr_copy_QKV.partition_D(sQ);
//    Tensor tQrQ = make_fragment_like(tQsQ);
//
//    Tensor tKgK = thr_copy_QKV.partition_S(gK);
//    Tensor tKsK = thr_copy_QKV.partition_D(sK);
//
//    Tensor tVgV = thr_copy_QKV.partition_S(gV);
//    Tensor tVsV = thr_copy_QKV.partition_D(sV);
//    Tensor tVrV = make_fragment_like(tVsV);
//
//    Tensor tdOgdO = thr_copy_QKV.partition_S(gdO);
//    Tensor tdOsdO = thr_copy_QKV.partition_D(sdO);
//    //Tensor tdOrdO = make_fragment_like(tdOsdO);
//
//    Tensor tdKsdK_copy = thr_copy_QKV.partition_S(sdK);
//    Tensor tdKgdK_copy = thr_copy_QKV.partition_D(gdK);
//
//    Tensor tdVsdV_copy = thr_copy_QKV.partition_S(sdV);
//    Tensor tdVgdV_copy = thr_copy_QKV.partition_D(gdV);
//
//
//
//    // S = QK^T
//    typename Kernel_traits::TiledMma_SdP tiled_mma_S;
//    ThrMMA thr_mma_S = tiled_mma_S.get_slice(threadIdx.x);
//    Tensor tSgQ = thr_mma_S.partition_A(gQ);
//    Tensor tSsQ = thr_mma_S.partition_A(sQ);
//    Tensor tSrQ = thr_mma_S.make_fragment_A(tSsQ);
//
//    Tensor tSgK = thr_mma_S.partition_B(gK);
//    Tensor tSsK = thr_mma_S.partition_B(sK);
//    Tensor tSrK = thr_mma_S.make_fragment_B(tSsK);
//
//    Tensor tSrS_float = partition_fragment_C(tiled_mma_S, Shape<Int<kBlockM>, Int<kBlockN>>{});
//    Tensor tSsP = thr_mma_S.partition_C(sP);
//
//    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQdOPdS{}, tiled_mma_S);
//    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(threadIdx.x);
//    auto tSsQ_copy_view = smem_thr_copy_Q.partition_S(sQ);
//    auto tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
//
//    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomKV{}, tiled_mma_S);
//    auto smem_thr_copy_K = smem_tiled_copy_K.get_slice(threadIdx.x);
//    auto tSsK_copy_view = smem_thr_copy_K.partition_S(sK);
//    auto tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);
//
//
//    // dP = dOV^T
//    typename Kernel_traits::TiledMma_SdP tiled_mma_dP;
//    ThrMMA thr_mma_dP = tiled_mma_dP.get_slice(threadIdx.x);
//
//    Tensor tdPgdO = thr_mma_dP.partition_A(gdO);
//    Tensor tdPsdO = thr_mma_dP.partition_A(sdO);
//    Tensor tdPrdO = thr_mma_dP.make_fragment_A(tdPsdO);
//
//    Tensor tdPgV = thr_mma_dP.partition_B(gV);
//    Tensor tdPsV = thr_mma_dP.partition_B(sV);
//    Tensor tdPrV = thr_mma_dP.make_fragment_B(tdPsV);
//
//    Tensor tdPrdP_float = partition_fragment_C(tiled_mma_dP, Shape<Int<kBlockM>, Int<kBlockN>>{});
//    Tensor tdPsdS = thr_mma_dP.partition_C(sdS);
//
//    auto smem_tiled_copy_dO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQdOPdS{}, tiled_mma_dP);
//    auto smem_thr_copy_dO = smem_tiled_copy_dO.get_slice(threadIdx.x);
//    auto tdPsdO_copy_view = smem_thr_copy_dO.partition_S(sdO);
//    auto tdPrdO_copy_view = smem_thr_copy_dO.retile_D(tdPrdO);
//
//    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomKV{}, tiled_mma_dP);
//    auto smem_thr_copy_V = smem_tiled_copy_V.get_slice(threadIdx.x);
//    auto tdPsV_copy_view = smem_thr_copy_V.partition_S(sV);
//    auto tdPrV_copy_view = smem_thr_copy_V.retile_D(tdPrV);
//
//
//
//    // dV += P^TdO
//    typename Kernel_traits::TiledMma_dKdV tiled_mma_dV;
//    ThrMMA thr_mma_dV = tiled_mma_dV.get_slice(threadIdx.x);
//
//    Tensor tdVsPt = thr_mma_dV.partition_A(sPt);
//    Tensor tdVrPt = thr_mma_dV.make_fragment_A(tdVsPt);
//
//    Tensor tdVsdOt = thr_mma_dV.partition_B(sdOt);
//    Tensor tdVrdOt = thr_mma_dV.make_fragment_B(tdVsdOt);
//
//    Tensor tdVrdV_float = partition_fragment_C(tiled_mma_dV, Shape<Int<kBlockN>, Int<kHeadDim>>{});
//    Tensor tdVsdV = thr_mma_dV.partition_C(sdV);
//
//    auto smem_tiled_copy_Pt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomPtdSt{}, tiled_mma_dV);
//    auto smem_thr_copy_Pt = smem_tiled_copy_Pt.get_slice(threadIdx.x);
//    auto tdVsPt_copy_view = smem_thr_copy_Pt.partition_S(sPt);
//    auto tdVrPt_copy_view = smem_thr_copy_Pt.retile_D(tdVrPt);
//
//    auto smem_tiled_copy_dOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomdOt{}, tiled_mma_dV);
//    auto smem_thr_copy_dOt = smem_tiled_copy_dOt.get_slice(threadIdx.x);
//    auto tdVsdOt_copy_view = smem_thr_copy_dOt.partition_S(sdOt);
//    auto tdVrdOt_copy_view = smem_thr_copy_dOt.retile_D(tdVrdOt);
//
//
//    // dK += dS^TQ
//    typename Kernel_traits::TiledMma_dKdV tiled_mma_dK;
//    ThrMMA thr_mma_dK = tiled_mma_dK.get_slice(threadIdx.x);
//    Tensor tdKsdSt = thr_mma_dK.partition_A(sdSt);
//    Tensor tdKrdSt = thr_mma_dK.make_fragment_A(tdKsdSt);
//
//    Tensor tdKsQt = thr_mma_dK.partition_B(sQt);
//    Tensor tdKrQt = thr_mma_dV.make_fragment_B(tdKsQt);
//
//    Tensor tdKrdK_float = partition_fragment_C(tiled_mma_dK, Shape<Int<kBlockN>, Int<kHeadDim>>{});
//    Tensor tdKsdK = thr_mma_dK.partition_C(sdK);
//
//    auto smem_tiled_copy_dSt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomPtdSt{}, tiled_mma_dK);
//    auto smem_thr_copy_dSt = smem_tiled_copy_dSt.get_slice(threadIdx.x);
//    auto tdKsdSt_copy_view = smem_thr_copy_dSt.partition_S(sdSt);
//    auto tdKrdSt_copy_view = smem_thr_copy_dSt.retile_D(tdKrdSt);
//
//    auto smem_tiled_copy_Qt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomQt{}, tiled_mma_dK);
//    auto smem_thr_copy_Qt = smem_tiled_copy_Qt.get_slice(threadIdx.x);
//    auto tdKsQt_copy_view = smem_thr_copy_dOt.partition_S(sQt);
//    auto tdKrQt_copy_view = smem_thr_copy_dOt.retile_D(tdKrQt);
//
//
//    // dQ = dSK
//    typename Kernel_traits::TiledMma_dQ tiled_mma_dQ;
//    ThrMMA thr_mma_dQ = tiled_mma_dQ.get_slice(threadIdx.x);
//    Tensor tdQsdS = thr_mma_dQ.partition_A(sdS);
//    Tensor tdQrdS = thr_mma_dQ.make_fragment_A(tdQsdS);
//    Tensor tdQsKt = thr_mma_dQ.partition_B(sKt);
//    Tensor tdQrKt = thr_mma_dQ.make_fragment_B(tdQsKt);
//
//    Tensor tdQrdQ_float = partition_fragment_C(tiled_mma_dQ, Shape<Int<kBlockM>, Int<kHeadDim>>{});
//    Tensor tdQsdQ = thr_mma_dQ.partition_C(sdQ);
//    Tensor tdQgdQ_float = thr_mma_dQ.partition_C(gdQ_float);
//
//    auto smem_tiled_copy_dS = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQdOPdS{}, tiled_mma_dQ);
//    auto smem_thr_copy_dS = smem_tiled_copy_dS.get_slice(threadIdx.x);
//    auto tdQsdS_copy_view = smem_thr_copy_dS.partition_S(sdS);
//    auto tdQrdS_copy_view = smem_thr_copy_dS.retile_D(tdQrdS);
//
//    auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomKt{}, tiled_mma_dQ);
//    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_slice(threadIdx.x);
//    auto tdQsKt_copy_view = smem_thr_copy_Kt.partition_S(sKt);
//    auto tdQrKt_copy_view = smem_thr_copy_Kt.retile_D(tdQrKt);
//
//
//
//
//
//    int Q_TILE_NO_MASK = 0;
//    int Q_TILE_MASK_START = 0;
//    int Q_TILE_MASK_END = 0;
//
//    if constexpr (Is_causal) {
//        // number of KV_TILES that does not need a mask
//        Q_TILE_NO_MASK = blockIdx.z + 1;
//        Q_TILE_MASK_START = blockIdx.z;
//        Q_TILE_MASK_END = Q_TILE_NO_MASK;
//
//    } else {
//        Q_TILE_NO_MASK = 0;
//    }
//
//
//    auto Q_TILE_MAX = size<3>(tSgQ);
//    auto QK_BLOCK_MAX = size<2>(tSsK);
//    auto PtdOt_BLOCK_MAX = size<2>(tdVsPt);
//    auto dSKt_BLOCK_MAX = size<2>(tdQsdS);
//    // load K, V, dK, dV tiles
//
//    copy(gmem_tiled_copy_QKV, tKgK, tKsK);
//    //copy(tSgK, tSsK);
//    copy(gmem_tiled_copy_QKV, tVgV, tVrV);
//
//
//    clear(tdVrdV_float);
//    clear(tdKrdK_float);
//
//
//    //copy(gmem_tiled_copy_QKV, tQgQ(_,_,_,Q_TILE_NO_MASK), tQrQ);
//    //copy(gmem_tiled_copy_QKV, tdOgdO(_,_,_,0), tdOrdO);
//
//    CUTE_NO_UNROLL
//    for (int q_tile = Q_TILE_NO_MASK; q_tile < Q_TILE_MAX; ++q_tile) {
//        copy(gmem_tiled_copy_QKV, tVrV, tVsV);
//
//        clear(tSrS_float);
//        clear(tdPrdP_float);
//
//        copy(gmem_tiled_copy_QKV, tQrQ, tQsQ);
//        //copy(gmem_tiled_copy_QKV, tdOrdO, tdOsdO);
//        // load gQ to sQ
//        copy(gmem_tiled_copy_QKV, tQgQ(_,_,_,q_tile), tQsQ);
//        copy(gmem_tiled_copy_QKV, tdOgdO(_,_,_,q_tile), tdOsdO);
//
//
//        __syncthreads();
//
//        // somehow pipelining gmem loads for both Q and dO use alot more registers which is slower
////         if (q_tile + 1 < Q_TILE_MAX) {
////             copy(gmem_tiled_copy_QKV, tQgQ(_,_,_,q_tile+1), tQrQ);
////             //copy(gmem_tiled_copy_QKV, tdOgdO(_,_,_,q_tile+1), tdOrdO);
////         }
//
//        // compute S=QK^T
//
//
////         copy(smem_tiled_copy_Q, tSsQ_copy_view(_,_,0), tSrQ_copy_view(_,_,0));
////         copy(smem_tiled_copy_K, tSsK_copy_view(_,_,0), tSrK_copy_view(_,_,0));
//
//        CUTE_UNROLL
//        for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
//            copy(smem_tiled_copy_Q, tSsQ_copy_view(_,_,qk_block), tSrQ_copy_view(_,_,qk_block));
//            copy(smem_tiled_copy_K, tSsK_copy_view(_,_,qk_block), tSrK_copy_view(_,_,qk_block));
//
////             int qk_block_next = (qk_block + 1) % QK_BLOCK_MAX;
////             copy(smem_tiled_copy_Q, tSsQ_copy_view(_,_,qk_block_next), tSrQ_copy_view(_,_,qk_block_next));
//            //copy(smem_tiled_copy_K, tSsK_copy_view(_,_,qk_block_next), tSrK_copy_view(_,_,qk_block_next));
//
//            copy(smem_tiled_copy_dO, tdPsdO_copy_view(_,_,qk_block), tdPrdO_copy_view(_,_,qk_block));
//            copy(smem_tiled_copy_V, tdPsV_copy_view(_,_,qk_block), tdPrV_copy_view(_,_,qk_block));
//
//            gemm(tiled_mma_S, tSrQ(_,_,qk_block), tSrK(_,_,qk_block), tSrS_float);
//            gemm(tiled_mma_dP, tdPrdO(_,_,qk_block), tdPrV(_,_,qk_block), tdPrdP_float);
//        }
//
//
//        //if (blockIdx.z == 0 && warp_id == 7 && lane_id == 31) {
//
//
//
////         gemm(tiled_mma_S, tSsQ, tSsK, tSrS_float);
////         gemm(tiled_mma_dP, tdPsdO, tdPsV, tdPrdP_float);
//
//
//        __syncthreads();
//
//        // load rL, rD from gmem to rmem
//        for (int i=0;i<2;i++) {
//            for (int j=0;j<2;j++) {
//                rL[i][j] = gL((warp_offset + thread_offset + 8 * j + 32 * i), q_tile);
//                rD[i][j] = gD((warp_offset + thread_offset + 8 * j + 32 * i), q_tile);
//            }
//        }
//
//        // rescale S
//        for (int i=0;i< tSrS_float.size();i ++ ) {
//            tSrS_float[i] *= 1.0f / sqrtf(kHeadDim);
//        }
//
//
//        Tensor tSrP_float = tSrS_float;
//        Tensor tdPrdS_float = tdPrdP_float;
//
//        // compute P = exp(S-l)
//        for (int i=0; i<2; i++) {
//            for (int j=0;j<2;j++) {
//                for (int k=0; k< tSrS_float(make_coord(_,j),i,_).size(); k++) {
//                    tSrP_float(make_coord(_,j),i,_)[k] = expf(tSrS_float(make_coord(_,j),i,_)[k] - rL[i][j]);
//                }
//            }
//        }
//
//
//
//        // compute dS = P \circ (dP - D)
//        // tS has the same mma layout as tdP
//        for (int i=0; i<2; i++) {
//            for (int j=0;j<2;j++) {
//                for (int k=0; k< tdPrdP_float(make_coord(_,j),i,_).size(); k++) {
//                    tdPrdS_float(make_coord(_,j),i,_)[k] = tSrP_float(make_coord(_,j),i,_)[k] * (tdPrdP_float(make_coord(_,j),i,_)[k] - rD[i][j]);
//                }
//            }
//        }
//
//        //convert P from fp32 to fp16
////         constexpr int num_element = decltype(size(tSrP_float))::value;
////
////         cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
////         auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tSrP_float.data()));
////
////         Tensor tSrP = make_tensor(make_rmem_ptr<half_t>(&frag), tSrP_float.layout());
//
//        Tensor tSrP = convert_type<half_t>(tSrP_float);
////
////
//        // convert dS from fp32 to fp16
////         constexpr int num_element_dS = decltype(size(tdPrdS_float))::value;
////
////         cutlass::NumericArrayConverter<half_t, float, num_element_dS> convert_op_dS;
////         auto frag_dS = convert_op_dS(*reinterpret_cast<const cutlass::Array<float, num_element_dS> *>(tdPrdS_float.data()));
////
////         Tensor tdPrdS = make_tensor(make_rmem_ptr<half_t>(&frag_dS), tdPrdS_float.layout());
//
//        Tensor tdPrdS = convert_type<half_t>(tdPrdS_float);
//
////
////
//        copy(gmem_tiled_copy_QKV, tVsV, tVrV);
//        __syncthreads();
//
////        if (thread(0)) {
////             //printf("q_tile = %d, warp_id = %d, lane_id = %d\n", q_tile, warp_id, lane_id);
////             //printf("tSrS_float\n");
////             print_tensor(tSrP);
////             //printf("====================\n");
////             //print_tensor(sdS(1, _));
////         }
//
//        copy(tSrP, tSsP);
//        copy(tdPrdS, tdPsdS);
////
////
//        __syncthreads();
//
////         if (thread(0) && q_tile == 0 ) {
////             printf("tdVsPt\n");
////             print_tensor(tdVsPt);
////             printf("tdVrPt\n");
////             print_tensor(tdVrPt);
////             printf("tdVsdOt\n");
////             print_tensor(tdVsdOt);
////             printf("tdVrdOt\n");
////             print_tensor(tdVrdOt);
////
////             printf("tdVsPt_copy_view\n");
////             print_tensor(tdVsPt_copy_view);
////             printf("tdVrPt_copy_view\n");
////             print_tensor(tdVrPt_copy_view);
////             printf("tdVsdOt_copy_view\n");
////             print_tensor(tdVsdOt_copy_view);
////             printf("tdVrdOt_copy_view\n");
////             print_tensor(tdVrdOt_copy_view);
////         }
//
//
//
//        // dV += P^TdO
//        CUTE_UNROLL
//        for (int ptdot_block = 0; ptdot_block < PtdOt_BLOCK_MAX; ptdot_block++) {
//            copy(smem_tiled_copy_dSt, tdVsPt_copy_view(_,_,ptdot_block), tdVrPt_copy_view(_,_,ptdot_block));
//            copy(smem_tiled_copy_Qt, tdVsdOt_copy_view(_,_,ptdot_block), tdVrdOt_copy_view(_,_,ptdot_block));
//
//            //copy(tdVsPt(_,_,ptdot_block), tdVrPt(_,_,ptdot_block));
//            //copy(tdVsdOt(_,_,ptdot_block), tdVrdOt(_,_,ptdot_block));
//            gemm(tiled_mma_dV, tdVrPt(_,_,ptdot_block), tdVrdOt(_,_,ptdot_block), tdVrdV_float);
//
//        }
//
//
//        //gemm(tiled_mma_dV, tdVsPt, tdVsdOt, tdVrdV_float);
////
//
//
//
//        // dK += dS^TQ
//
//        CUTE_UNROLL
//        for (int ptdot_block = 0; ptdot_block < PtdOt_BLOCK_MAX; ptdot_block++) {
//            copy(smem_tiled_copy_dSt, tdKsdSt_copy_view(_,_,ptdot_block), tdKrdSt_copy_view(_,_,ptdot_block));
//            copy(smem_tiled_copy_Qt, tdKsQt_copy_view(_,_,ptdot_block), tdKrQt_copy_view(_,_,ptdot_block));
//
//            //copy(tdVsPt(_,_,ptdot_block), tdVrPt(_,_,ptdot_block));
//            //copy(tdVsdOt(_,_,ptdot_block), tdVrdOt(_,_,ptdot_block));
//            gemm(tiled_mma_dK, tdKrdSt(_,_,ptdot_block), tdKrQt(_,_,ptdot_block), tdKrdK_float);
//
//        }
//
//        //gemm(tiled_mma_dK, tdKsdSt, tdKsQt, tdKrdK_float);
//
//        CUTE_UNROLL
//        for (int dskt_block = 0; dskt_block < dSKt_BLOCK_MAX; dskt_block++) {
//            copy(smem_tiled_copy_dS, tdQsdS_copy_view(_,_,dskt_block), tdQrdS_copy_view(_,_,dskt_block));
//            copy(smem_tiled_copy_Kt, tdQsKt_copy_view(_,_,dskt_block), tdQrKt_copy_view(_,_,dskt_block));
//
//
//            gemm(tiled_mma_dQ, tdQrdS(_,_,dskt_block), tdQrKt(_,_,dskt_block), tdQrdQ_float);
////                 if (blockIdx.z == 0 && warp_id==4 && lane_id==0) {
////                     print_tensor(tdQrdQ_float);
////                 }
//
//        }
//
//        for (int i=0;i< tdQrdQ_float.size();i ++ ) {
//            tdQrdQ_float[i] *= 1.0f / sqrtf(kHeadDim);
//        }
//
////
////         if (thread0)
////         print_tensor(tdQrdQ_float);
//
//
//        // for (int i = 0; i < size(acc_dq); ++i) { atomicAdd(&tdQgdQaccum(i), acc_dq(i)); }
//        for (int i=0; i< tdQrdQ_float.size(); i++) {
//            atomicAdd(&tdQgdQ_float(_,_,_,q_tile)[i], tdQrdQ_float[i]);
//        }
//
//
//
//        __syncthreads();
//
//    }
//
////     To compute the mask, imagine we are working in a 64 x 64 row-major matrix.
////     We want to compute a map
////     (warp_id, lane_id) -> offset in a 64 x 64 row major matrix -> 2d coordinates
////     To derive the first map we can partition a 64 x 64 row-major matrix using the bwd mma
////     and print out the layout of each thread partition.
////     For hdim = 128, the layout is ((_2,_2),_2,_2):((_1,_512),_2048,_32)
////     The starting offset of each thread is given by
////     (warp_id / 4) * 64 * 16 + (warp_id % 4) * 8 + (lane_id / 4) * 64 + (lane_id % 4) * 2
////     the offset is simply offset_start + dot product between layout and stride
////     The second map is simply offset -> (offset / 64, offset % 64)
//
//
//    if (Is_causal) {
//        copy(gmem_tiled_copy_QKV, tQgQ(_,_,_,Q_TILE_MASK_START), tQrQ);
//        //copy(gmem_tiled_copy_QKV, tdOgdO(_,_,_,0), tdOrdO);
//
//        CUTE_NO_UNROLL
//        for (int q_tile = Q_TILE_MASK_START; q_tile < Q_TILE_MASK_END; ++q_tile) {
//            copy(gmem_tiled_copy_QKV, tVrV, tVsV);
//
//            clear(tSrS_float);
//            clear(tdPrdP_float);
//
//            copy(gmem_tiled_copy_QKV, tQrQ, tQsQ);
//            //copy(gmem_tiled_copy_QKV, tdOrdO, tdOsdO);
//            // load gQ to sQ
//            //copy(gmem_tiled_copy_QKV, tQgQ(_,_,_,q_tile), tQsQ);
//            copy(gmem_tiled_copy_QKV, tdOgdO(_,_,_,q_tile), tdOsdO);
//
//
//            __syncthreads();
//
//            // somehow pipelining gmem loads for both Q and dO use alot more registers which is slower
//            if (q_tile + 1 < Q_TILE_MASK_END) {
//                copy(gmem_tiled_copy_QKV, tQgQ(_,_,_,q_tile+1), tQrQ);
//                //copy(gmem_tiled_copy_QKV, tdOgdO(_,_,_,q_tile+1), tdOrdO);
//            }
//
//            // compute S=QK^T
//
//
//    //         copy(smem_tiled_copy_Q, tSsQ_copy_view(_,_,0), tSrQ_copy_view(_,_,0));
//    //         copy(smem_tiled_copy_K, tSsK_copy_view(_,_,0), tSrK_copy_view(_,_,0));
//
//            CUTE_UNROLL
//            for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
//                copy(smem_tiled_copy_Q, tSsQ_copy_view(_,_,qk_block), tSrQ_copy_view(_,_,qk_block));
//                copy(smem_tiled_copy_K, tSsK_copy_view(_,_,qk_block), tSrK_copy_view(_,_,qk_block));
//
//    //             int qk_block_next = (qk_block + 1) % QK_BLOCK_MAX;
//    //             copy(smem_tiled_copy_Q, tSsQ_copy_view(_,_,qk_block_next), tSrQ_copy_view(_,_,qk_block_next));
//                //copy(smem_tiled_copy_K, tSsK_copy_view(_,_,qk_block_next), tSrK_copy_view(_,_,qk_block_next));
//
//                copy(smem_tiled_copy_dO, tdPsdO_copy_view(_,_,qk_block), tdPrdO_copy_view(_,_,qk_block));
//                copy(smem_tiled_copy_V, tdPsV_copy_view(_,_,qk_block), tdPrV_copy_view(_,_,qk_block));
//
//                gemm(tiled_mma_S, tSrQ(_,_,qk_block), tSrK(_,_,qk_block), tSrS_float);
//                gemm(tiled_mma_dP, tdPrdO(_,_,qk_block), tdPrV(_,_,qk_block), tdPrdP_float);
//            }
//
//
//
//
//
//    //         gemm(tiled_mma_S, tSsQ, tSsK, tSrS_float);
//    //         gemm(tiled_mma_dP, tdPsdO, tdPsV, tdPrdP_float);
//
//
//            __syncthreads();
//
//            // load rL, rD from gmem to rmem
//            for (int i=0;i<2;i++) {
//                for (int j=0;j<2;j++) {
//                    rL[i][j] = gL((warp_offset + thread_offset + 8 * j + 32 * i), q_tile);
//                    rD[i][j] = gD((warp_offset + thread_offset + 8 * j + 32 * i), q_tile);
//                }
//            }
//
//            // rescale S
//            for (int i=0;i< tSrS_float.size();i ++ ) {
//                tSrS_float[i] *= 1.0f / sqrtf(kHeadDim);
//            }
//
//
//            //apply mask
//            int row_offset = (warp_id % 2) * 16 + (lane_id / 4);
//            int col_offset = (warp_id / 2) * 8 + (lane_id % 4) * 2;
//            for (int i=0; i<2; i++) {
//                for (int j=0;j<2;j++) {
//                    for (int k=0;k<2;k++) {
//                        for (int l=0;l<2;l++) {
//                            int row = row_offset + 8 * j + 32 * k;
//                            int col = col_offset + i + 32 * l;
//                            if (row < col) {
//                                tSrS_float(make_coord(i,j),k,l) = -FLT_MAX;
//                                tdPrdP_float(make_coord(i,j),k,l) = 0;
//                            }
//                            // print("((%d, %d), %d, %d)\n", i, j, k, l);
//                            // print("row = %d, col = %d\n", row_offset + 8 * j + 32 * k, col_offset + i + 32 * l );
//                            // print("%d\n", 64 * (row_offset + 8 * j + 32 * k) + col_offset + i + 32 * l);
//                            // printf("%d\n", tc(make_coord(i,j),k,l));
//                            // print("====================\n");
//                        }
//                    }
//                }
//            }
//
////             if (blockIdx.z == 0 && warp_id == 6 && lane_id == 4) {
////                 printf("kv_tile = %d, warp_id = %d, lane_id = %d\n", q_tile, warp_id, lane_id);
////                 printf("tSrS_float\n");
////                 print_tensor(tSrS_float(make_coord(_, _), 0, 1));
////                 printf("====================\n");
////                 //print_tensor(sdS(1, _));
////             }
//
//
//            Tensor tSrP_float = tSrS_float;
//            Tensor tdPrdS_float = tdPrdP_float;
//
//            // compute P = exp(S-l)
//            for (int i=0; i<2; i++) {
//                for (int j=0;j<2;j++) {
//                    for (int k=0; k< tSrS_float(make_coord(_,j),i,_).size(); k++) {
//                        tSrP_float(make_coord(_,j),i,_)[k] = expf(tSrS_float(make_coord(_,j),i,_)[k] - rL[i][j]);
//                    }
//                }
//            }
//    //
//            // compute dS = P \circ (dP - D)
//            // tS has the same mma layout as tdP
//            for (int i=0; i<2; i++) {
//                for (int j=0;j<2;j++) {
//                    for (int k=0; k< tdPrdP_float(make_coord(_,j),i,_).size(); k++) {
//                        tdPrdS_float(make_coord(_,j),i,_)[k] = tSrP_float(make_coord(_,j),i,_)[k] * (tdPrdP_float(make_coord(_,j),i,_)[k] - rD[i][j]);
//                    }
//                }
//            }
//
//            //convert P from fp32 to fp16
////             constexpr int num_element = decltype(size(tSrP_float))::value;
////
////             cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
////             auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tSrP_float.data()));
////
////             Tensor tSrP = make_tensor(make_rmem_ptr<half_t>(&frag), tSrP_float.layout());
//            Tensor tSrP = convert_type<half_t>(tSrP_float);
//    //
//    //
//            // convert dS from fp32 to fp16
////             constexpr int num_element_dS = decltype(size(tdPrdS_float))::value;
////
////             cutlass::NumericArrayConverter<half_t, float, num_element_dS> convert_op_dS;
////             auto frag_dS = convert_op_dS(*reinterpret_cast<const cutlass::Array<float, num_element_dS> *>(tdPrdS_float.data()));
////
////             Tensor tdPrdS = make_tensor(make_rmem_ptr<half_t>(&frag_dS), tdPrdS_float.layout());
//
//            Tensor tdPrdS = convert_type<half_t>(tdPrdS_float);
//
//    //
//    //
//            copy(gmem_tiled_copy_QKV, tVsV, tVrV);
//            __syncthreads();
//
//            copy(tSrP, tSsP);
//            copy(tdPrdS, tdPsdS);
//    //
//    //
//            __syncthreads();
//
//
////             if (thread(0)) {
////                 printf("q_tile = %d\n", q_tile);
////                 print_tensor(sPt(62, _));
////                 printf("tdVsPt\n");
////                 print_tensor(tdVsPt);
////                 printf("tdVrPt\n");
////                 print_tensor(tdVrPt);
////                 printf("tdVsdOt\n");
////                 print_tensor(tdVsdOt);
////                 printf("tdVrdOt\n");
////                 print_tensor(tdVrdOt);
////
////                 printf("tdVsPt_copy_view\n");
////                 print_tensor(tdVsPt_copy_view);
////                 printf("tdVrPt_copy_view\n");
////                 print_tensor(tdVrPt_copy_view);
////                 printf("tdVsdOt_copy_view\n");
////                 print_tensor(tdVsdOt_copy_view);
////                 printf("tdVrdOt_copy_view\n");
////                 print_tensor(tdVrdOt_copy_view);
////             }
//
//
//
//            // dV += P^TdO
//            CUTE_UNROLL
//            for (int ptdot_block = 0; ptdot_block < PtdOt_BLOCK_MAX; ptdot_block++) {
//                copy(smem_tiled_copy_dSt, tdVsPt_copy_view(_,_,ptdot_block), tdVrPt_copy_view(_,_,ptdot_block));
//                copy(smem_tiled_copy_Qt, tdVsdOt_copy_view(_,_,ptdot_block), tdVrdOt_copy_view(_,_,ptdot_block));
//
//                //copy(tdVsPt(_,_,ptdot_block), tdVrPt(_,_,ptdot_block));
//                //copy(tdVsdOt(_,_,ptdot_block), tdVrdOt(_,_,ptdot_block));
//                gemm(tiled_mma_dV, tdVrPt(_,_,ptdot_block), tdVrdOt(_,_,ptdot_block), tdVrdV_float);
//
//            }
//
//
//            //gemm(tiled_mma_dV, tdVsPt, tdVsdOt, tdVrdV_float);
//    //
//
//
//
//            // dK += dS^TQ
//
//            CUTE_UNROLL
//            for (int ptdot_block = 0; ptdot_block < PtdOt_BLOCK_MAX; ptdot_block++) {
//                copy(smem_tiled_copy_dSt, tdKsdSt_copy_view(_,_,ptdot_block), tdKrdSt_copy_view(_,_,ptdot_block));
//                copy(smem_tiled_copy_Qt, tdKsQt_copy_view(_,_,ptdot_block), tdKrQt_copy_view(_,_,ptdot_block));
//
//                //copy(tdVsPt(_,_,ptdot_block), tdVrPt(_,_,ptdot_block));
//                //copy(tdVsdOt(_,_,ptdot_block), tdVrdOt(_,_,ptdot_block));
//                gemm(tiled_mma_dK, tdKrdSt(_,_,ptdot_block), tdKrQt(_,_,ptdot_block), tdKrdK_float);
//
//            }
//
//            //gemm(tiled_mma_dK, tdKsdSt, tdKsQt, tdKrdK_float);
//
//
//            __syncthreads();
//
//        }
//    }
//
//
//    // dV
////     constexpr int num_element = decltype(size(tdVrdV_float))::value;
////
////     cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
////     auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tdVrdV_float.data()));
////
////     Tensor tdVrdV = make_tensor(make_rmem_ptr<half_t>(&frag), tdVrdV_float.layout());
//
//    Tensor tdVrdV = convert_type<half_t>(tdVrdV_float);
//
//    copy(tdVrdV, tdVsdV);
////
////
//    // dK
//
//    // rescale by head dim
//    for (int i=0;i< tdKrdK_float.size();i ++ ) {
//        tdKrdK_float[i] *= 1.0f / sqrtf(kHeadDim);
//    }
//
//
////     constexpr int num_element_dK = decltype(size(tdKrdK_float))::value;
////
////     cutlass::NumericArrayConverter<half_t, float, num_element_dK> convert_op_dK;
////     auto frag_dK = convert_op_dK(*reinterpret_cast<const cutlass::Array<float, num_element_dK> *>(tdKrdK_float.data()));
////
////     Tensor tdKrdK = make_tensor(make_rmem_ptr<half_t>(&frag_dK), tdKrdK_float.layout());
//
//    Tensor tdKrdK = convert_type<half_t>(tdKrdK_float);
//
//    copy(tdKrdK, tdKsdK);
//
//    __syncthreads();
//
////     copy(tdQrdQ, tdQsdQ);
////
////     __syncthreads();
////
////     copy(gmem_tiled_copy_QKV, tdQsdQ_copy, tdQgdQ_copy);
//    copy(gmem_tiled_copy_QKV, tdKsdK_copy, tdKgdK_copy);
//    copy(gmem_tiled_copy_QKV, tdVsdV_copy, tdVgdV_copy);
//
//}


template <typename Kernel_traits, bool Is_causal>
inline __device__ void compute_dq(
    half_t * __restrict__ q_ptr,
    half_t * __restrict__ k_ptr,
    half_t * __restrict__ v_ptr,
    float *__restrict__ l_ptr,
    float * __restrict__ d_ptr,
    half_t * __restrict__ do_ptr,
    half_t* __restrict__ dq_ptr,
    int batch_size, int seq_len, int num_heads, int head_dim, int is_causal
) {

    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    compute_dq_1rowblock<Kernel_traits, Is_causal>(q_ptr,
                                                    k_ptr,
                                                    v_ptr,
                                                    l_ptr,
                                                    d_ptr,
                                                    do_ptr,
                                                    dq_ptr,
                                                    batch_size,
                                                    seq_len,
                                                    num_heads,
                                                    head_dim,
                                                    is_causal,
                                                    bidb,
                                                    bidh,
                                                    m_block);

}

template <typename Kernel_traits, bool Is_causal>
inline __device__ void compute_dk_dv(
    half_t * __restrict__ q_ptr,
    half_t * __restrict__ k_ptr,
    half_t * __restrict__ v_ptr,
    float * __restrict__ l_ptr,
    float * __restrict__ d_ptr,
    half_t * __restrict__ do_ptr,
    half_t* __restrict__ dk_ptr,
    half_t* __restrict__ dv_ptr,
    int batch_size, int seq_len, int num_heads, int head_dim, int is_causal
) {
    const int n_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    compute_dk_dv_1colblock<Kernel_traits, Is_causal>(q_ptr,
                                                    k_ptr,
                                                    v_ptr,
                                                    l_ptr,
                                                    d_ptr,
                                                    do_ptr,
                                                    dk_ptr,
                                                    dv_ptr,
                                                    batch_size,
                                                    seq_len,
                                                    num_heads,
                                                    head_dim,
                                                    is_causal,
                                                    bidb,
                                                    bidh,
                                                    n_block);

}