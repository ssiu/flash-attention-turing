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
#include "mask.h"

using namespace cute;





template <typename Kernel_traits, bool Is_causal, bool Is_even_MN>
inline __device__ void compute_dq_1rowblock(
    const half_t * __restrict__ q_ptr,
    const half_t * __restrict__ k_ptr,
    const half_t * __restrict__ v_ptr,
    float *__restrict__ l_ptr,
    float * __restrict__ d_ptr,
    const half_t * __restrict__ do_ptr,
    half_t* __restrict__ dq_ptr,
    int * __restrict__ cu_seqlens_q,
    int * __restrict__ cu_seqlens_k,
    int batch_size,
    int seqlen_q, 
    int seqlen_k, 
    int num_heads, 
    int num_heads_k,
    int h_h_k_ratio,
    int head_dim, 
    float softmax_scale,
    int is_causal,
    int bidb,
    int bidh,
    int m_block
)
{
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;


    const int max_seqlen_q = seqlen_q;
    const int max_seqlen_k = seqlen_k;
    const BlockInfo binfo(max_seqlen_q, max_seqlen_k, bidb, cu_seqlens_q, cu_seqlens_k);

    seqlen_q = binfo.actual_seqlen_q;
    seqlen_k = binfo.actual_seqlen_k;


    if (m_block * kBlockM >= seqlen_q) {
        return;
    }
    

    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;

    // for 8 warps, the 32x32 tiled mma is like
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

    Tensor mQ = make_tensor(make_gmem_ptr(q_ptr + binfo.q_offset(num_heads * head_dim, bidb)),
                            make_shape(seqlen_q, num_heads, head_dim),
                            make_stride(num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));

    // K

    Tensor mK = make_tensor(make_gmem_ptr(k_ptr + binfo.k_offset(num_heads_k * head_dim, bidb)),
                            make_shape(seqlen_k, num_heads_k, head_dim),
                            make_stride(num_heads_k * head_dim, head_dim, Int<1>{}));

    Tensor gK = local_tile(mK(_, bidh / h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));

    // V

    Tensor mV = make_tensor(make_gmem_ptr(v_ptr + binfo.k_offset(num_heads_k * head_dim, bidb)),
                            make_shape(seqlen_k, num_heads_k, head_dim),
                            make_stride(num_heads_k * head_dim, head_dim, Int<1>{}));

    Tensor gV = local_tile(mV(_, bidh / h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));


    // L = m + log l
    Tensor mL = make_tensor(make_gmem_ptr(l_ptr),
                             make_shape(batch_size, num_heads, max_seqlen_q),
                             make_stride(max_seqlen_q * num_heads,  max_seqlen_q, Int<1>{}));

    Tensor gL = local_tile(mL(bidb, bidh, _), Shape<Int<kBlockM>>{},
                           make_coord(m_block));


    Tensor mD = make_tensor(make_gmem_ptr(d_ptr),
                             make_shape(batch_size, num_heads, max_seqlen_q),
                             make_stride(max_seqlen_q * num_heads,  max_seqlen_q, Int<1>{}));

    Tensor gD = local_tile(mD(bidb, bidh, _), Shape<Int<kBlockM>>{},
                           make_coord(m_block));

    // dO

    Tensor mdO = make_tensor(make_gmem_ptr(do_ptr + binfo.q_offset(num_heads * head_dim, bidb)),
                             make_shape(seqlen_q, num_heads, head_dim),
                             make_stride(num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gdO = local_tile(mdO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));


    // dQ

    Tensor mdQ = make_tensor(make_gmem_ptr(dq_ptr + binfo.q_offset(num_heads * head_dim, bidb)),
                            make_shape(seqlen_q, num_heads, head_dim),
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
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    //int thread_row = warp_id * 16 + lane_id / 4;
    const int warp_offset = (warp_id % 2) * 16;
    const int thread_offset = lane_id / 4;

    const int global_row_offset = m_block * kBlockM;


    float rL[2][2] = {0};
    float rD[2][2] = {0};


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


    const int n_block_min = 0;
    const int m_block_max = ceil_div(seqlen_q, kBlockM);
    int n_block_max = ceil_div(seqlen_k, kBlockN);

    int n_masking_steps = (!Is_causal)
        ? 1
        : ((Is_even_MN && Is_causal) ? ceil_div(kBlockM, kBlockN) : ceil_div(kBlockM, kBlockN) + 1);

    if constexpr(Is_causal) {

        n_block_max = fmaxf(0, ceil_div((m_block + 1) * kBlockM + seqlen_k - seqlen_q, kBlockN));
        n_masking_steps = fminf(n_masking_steps, n_block_max);
    }


    // if seqlen_q > seqlen_k we exit early for the blocks with rows that are fully masked
    if (n_block_max == 0) {return;}



    //auto KV_TILE_MAX = size<3>(tSgK);
    auto QK_BLOCK_MAX = size<2>(tSsK);
    auto dSKt_BLOCK_MAX = size<2>(tdQsdS);


    // load K, V, dK, dV tiles

    int n_block = n_block_max - 1;


//    Mask<Is_causal> accum_s_mask(seqlen_q, seqlen_k);

    masked_copy<Is_even_MN>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, warp_id, lane_id, 
        seqlen_q - m_block * kBlockM, 
        /*clear_D=*/true);
    masked_copy<Is_even_MN>(
        gmem_tiled_copy_QKV, tdOgdO, tdOsdO, warp_id, lane_id, 
        seqlen_q - m_block * kBlockM, 
        /*clear_D=*/true);



//    masked_copy<Is_even_MN>(gmem_tiled_copy_QKV, tKgK(_,_,_,n_block), tKrK, warp_id, lane_id, seqlen_k - n_block * kBlockN);
//    masked_copy<Is_even_MN>(gmem_tiled_copy_QKV, tVgV(_,_,_,n_block), tVrV, warp_id, lane_id, seqlen_k - n_block * kBlockN);


    Mask<Is_causal> accum_SdP_mask(seqlen_q, seqlen_k);


    clear(tdQrdQ_float);

            // load rL, rD from gmem to rmem


    for (int i=0;i<2;i++) {
        for (int j=0;j<2;j++) {
            int global_row = global_row_offset + warp_offset + thread_offset + 8 * j + 32 * i;
            if (global_row < seqlen_q) {
                rL[i][j] = gL(warp_offset + thread_offset + 8 * j + 32 * i);
                rD[i][j] = gD(warp_offset + thread_offset + 8 * j + 32 * i);
//                    print("bwd, thread_id = %d, thread_row = %d, rL = %.8e, rD = %.8e\n", threadIdx.x, warp_offset + thread_offset + 8 * j + 32 * i, rL[i][j], rD[i][j]);
            }
        }
    }

    CUTE_NO_UNROLL
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        clear(tSrS_float);
        clear(tdPrdP_float);

        masked_copy<Is_even_MN>(
            gmem_tiled_copy_QKV, tKgK(_,_,_,n_block), tKrK, warp_id, lane_id, 
            seqlen_k - n_block * kBlockN, 
            /*clear_D=*/true);
        masked_copy<Is_even_MN>(
            gmem_tiled_copy_QKV, tVgV(_,_,_,n_block), tVrV, warp_id, lane_id, 
            seqlen_k - n_block * kBlockN, 
            /*clear_D=*/true);
        copy(gmem_tiled_copy_QKV, tKrK, tKsK);
        copy(gmem_tiled_copy_QKV, tVrV, tVsV);

        __syncthreads();


//        if (n_block > n_block_min) {
//            masked_copy<Is_even_MN>(gmem_tiled_copy_QKV, tKgK(_,_,_,n_block - 1), tKrK, warp_id, lane_id, seqlen_k - n_block * kBlockN);
//            masked_copy<Is_even_MN>(gmem_tiled_copy_QKV, tVgV(_,_,_,n_block - 1), tVrV, warp_id, lane_id, seqlen_k - n_block * kBlockN);
//        }



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


        // rescale S
        for (int i=0;i< tSrS_float.size();i ++ ) {
            tSrS_float[i] *= softmax_scale;
        }


        accum_SdP_mask.template apply_mask_bwd_dq<Is_causal, Is_even_MN>(
            tSrS_float, tdPrdP_float, warp_id, lane_id, m_block, n_block, seqlen_q, seqlen_k, kBlockM, kBlockN, head_dim
        );



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


        Tensor tdPrdS = convert_type<half_t>(tdPrdS_float);

        // copy(tSrP, tSsP);
        copy(tdPrdS, tdPsdS);


        __syncthreads();

        // dQ += dSK


        CUTE_UNROLL
        for (int dskt_block = 0; dskt_block < dSKt_BLOCK_MAX; dskt_block++) {
            copy(smem_tiled_copy_dS, tdQsdS_copy_view(_,_,dskt_block), tdQrdS_copy_view(_,_,dskt_block));
            copy(smem_tiled_copy_Kt, tdQsKt_copy_view(_,_,dskt_block), tdQrKt_copy_view(_,_,dskt_block));


            gemm(tiled_mma_dQ, tdQrdS(_,_,dskt_block), tdQrKt(_,_,dskt_block), tdQrdQ_float);


        }


        __syncthreads();

    }



    CUTE_NO_UNROLL
    for (; n_block >= n_block_min; --n_block) {
        
        const int global_row_offset = m_block * kBlockM;


        clear(tSrS_float);
        clear(tdPrdP_float);

        copy(gmem_tiled_copy_QKV, tKgK(_,_,_,n_block), tKrK);
        copy(gmem_tiled_copy_QKV, tVgV(_,_,_,n_block), tVrV);
        copy(gmem_tiled_copy_QKV, tKrK, tKsK);
        copy(gmem_tiled_copy_QKV, tVrV, tVsV);


        __syncthreads();


//        if (n_block > n_block_min) {
//            copy(gmem_tiled_copy_QKV, tKgK(_,_,_,n_block - 1), tKrK);
//            copy(gmem_tiled_copy_QKV, tVgV(_,_,_,n_block - 1), tVrV);
//        }



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



        // rescale S
        for (int i=0;i< tSrS_float.size();i ++ ) {
            tSrS_float[i] *= softmax_scale;
        }

        for (int i=0;i<2;i++) {
            for (int j=0;j<2;j++) {
                const int global_row = global_row_offset + warp_offset + thread_offset + 8 * j + 32 * i;
                if (global_row < seqlen_q) {
                    rL[i][j] = gL(warp_offset + thread_offset + 8 * j + 32 * i);
                    rD[i][j] = gD(warp_offset + thread_offset + 8 * j + 32 * i);
                }
            }
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



        Tensor tdPrdS = convert_type<half_t>(tdPrdS_float);

        //copy(tSrP, tSsP);
        copy(tdPrdS, tdPsdS);

        __syncthreads();



        // dQ += dSK
 



        CUTE_UNROLL
        for (int dskt_block = 0; dskt_block < dSKt_BLOCK_MAX; dskt_block++) {
            copy(smem_tiled_copy_dS, tdQsdS_copy_view(_,_,dskt_block), tdQrdS_copy_view(_,_,dskt_block));
            copy(smem_tiled_copy_Kt, tdQsKt_copy_view(_,_,dskt_block), tdQrKt_copy_view(_,_,dskt_block));


            gemm(tiled_mma_dQ, tdQrdS(_,_,dskt_block), tdQrKt(_,_,dskt_block), tdQrdQ_float);

        }
 
        __syncthreads();

    }
    

    // rescale by head dim
    for (int i=0;i< tdQrdQ_float.size();i ++ ) {
        tdQrdQ_float[i] *= softmax_scale;
    }



    Tensor tdQrdQ = convert_type<half_t>(tdQrdQ_float);



    copy(tdQrdQ, tdQsdQ);


    __syncthreads();



    masked_copy<Is_even_MN>(
        gmem_tiled_copy_QKV, tdQsdQ_copy, tdQgdQ_copy, warp_id, lane_id, 
        seqlen_q - m_block * kBlockM, 
        /*clear_D=*/false);


}



template <typename Kernel_traits, bool Is_causal, bool Is_even_MN>
inline __device__ void compute_dk_dv_1colblock(
    half_t * __restrict__ q_ptr,
    half_t * __restrict__ k_ptr,
    half_t * __restrict__ v_ptr,
    float * __restrict__ l_ptr,
    float * __restrict__ d_ptr,
    half_t * __restrict__ do_ptr,
    half_t* __restrict__ dk_ptr,
    half_t* __restrict__ dv_ptr,
    int * __restrict__ cu_seqlens_q,
    int * __restrict__ cu_seqlens_k,
    int batch_size, int seqlen_q, int seqlen_k, int num_heads, int num_heads_k, int h_h_k_ratio, int head_dim, float softmax_scale, int is_causal,
    int bidb, int bidh, int n_block
)
{   
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    const int max_seqlen_q = seqlen_q;
    const int max_seqlen_k = seqlen_k;
    const BlockInfo binfo(max_seqlen_q, max_seqlen_k, bidb, cu_seqlens_q, cu_seqlens_k);
    seqlen_q = binfo.actual_seqlen_q;
    seqlen_k = binfo.actual_seqlen_k;
    
    if (n_block * kBlockN >= seqlen_k) {
        return;
    }

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

    Tensor mQ = make_tensor(make_gmem_ptr(q_ptr + binfo.q_offset(num_heads * head_dim, bidb)),
                            make_shape(seqlen_q, num_heads, head_dim),
                            make_stride(num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(_, 0));

    // K

    Tensor mK = make_tensor(make_gmem_ptr(k_ptr + binfo.k_offset(num_heads_k * head_dim, bidb)),
                            make_shape(seqlen_k, num_heads_k, head_dim),
                            make_stride(num_heads_k * head_dim, head_dim, Int<1>{}));

    Tensor gK = local_tile(mK(_, bidh / h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(n_block, 0));

    // V

    Tensor mV = make_tensor(make_gmem_ptr(v_ptr+ binfo.k_offset(num_heads_k * head_dim, bidb)),
                            make_shape(seqlen_k, num_heads_k, head_dim),
                            make_stride(num_heads_k * head_dim, head_dim, Int<1>{}));

    Tensor gV = local_tile(mV(_, bidh / h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(n_block, 0));

    // L = m + log l
    Tensor mL = make_tensor(make_gmem_ptr(l_ptr),
                             make_shape(batch_size, num_heads, max_seqlen_q),
                             make_stride(max_seqlen_q * num_heads,  max_seqlen_q, Int<1>{}));

    Tensor gL = local_tile(mL(bidb, bidh, _), Shape<Int<kBlockM>>{},
                           make_coord(_));


    Tensor mD = make_tensor(make_gmem_ptr(d_ptr),
                             make_shape(batch_size, num_heads, max_seqlen_q),
                             make_stride(max_seqlen_q * num_heads,  max_seqlen_q, Int<1>{}));

    Tensor gD = local_tile(mD(bidb, bidh, _), Shape<Int<kBlockM>>{},
                           make_coord(_));

    // dO


    Tensor mdO = make_tensor(make_gmem_ptr(do_ptr+ binfo.q_offset(num_heads * head_dim, bidb)),
                             make_shape(seqlen_q, num_heads, head_dim),
                             make_stride(num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gdO = local_tile(mdO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(_, 0));
    // dV

    Tensor mdV = make_tensor(make_gmem_ptr(dv_ptr + binfo.k_offset(num_heads * head_dim, bidb)),
                            make_shape(seqlen_k, num_heads, head_dim),
                            make_stride(num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gdV = local_tile(mdV(_, bidh, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(n_block, 0));
    // dK

    Tensor mdK = make_tensor(make_gmem_ptr(dk_ptr + binfo.k_offset(num_heads * head_dim, bidb)),
                            make_shape(seqlen_k, num_heads, head_dim),
                            make_stride(num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gdK = local_tile(mdK(_, bidh, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
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
    Tensor tdOrdO = make_fragment_like(tdOsdO);

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





    // auto Q_TILE_MAX = size<3>(tSgQ);


    int m_block_min = 0;  
    int m_block_max = ceil_div(seqlen_q, kBlockM); 
    const int n_block_max = ceil_div(seqlen_k, kBlockN); 



    int n_masking_steps = (!Is_causal)
        ? 0
        : ((Is_even_MN && Is_causal) ? 1 : 2);



    if constexpr(Is_causal) {
        m_block_min = fmaxf(m_block_min, (n_block * kBlockN + seqlen_q - seqlen_k) / kBlockM);
        n_masking_steps = fminf(n_masking_steps, m_block_max);

    }

    auto QK_BLOCK_MAX = size<2>(tSsK);
    auto PtdOt_BLOCK_MAX = size<2>(tdVsPt);

    int m_block = m_block_min;


    
    masked_copy<Is_even_MN>(
        gmem_tiled_copy_QKV, tKgK, tKsK, warp_id, lane_id, 
        seqlen_k - n_block * kBlockN, 
        /*clear_D=*/true);
    masked_copy<Is_even_MN>(
        gmem_tiled_copy_QKV, tVgV, tVrV, warp_id, lane_id, 
        seqlen_k - n_block * kBlockN, 
        /*clear_D=*/true);


//    copy(gmem_tiled_copy_QKV, tQgQ(_,_,_,m_block_min), tQrQ);
    //copy(gmem_tiled_copy_QKV, tdOgdO(_,_,_,0), tdOrdO);
    Mask<Is_causal> accum_SdP_mask(seqlen_q, seqlen_k);

    clear(tdVrdV_float);
    clear(tdKrdK_float);

    CUTE_NO_UNROLL

    for (int masking_step = 0; masking_step < n_masking_steps && m_block < m_block_max; ++masking_step, ++m_block) {
        // copy(gmem_tiled_copy_QKV, tVrV, tVsV);
        copy(gmem_tiled_copy_QKV, tVrV, tVsV);



        clear(tSrS_float);
        clear(tdPrdP_float);


        // load gQ to sQ

        masked_copy<Is_even_MN>(
            gmem_tiled_copy_QKV, tQgQ(_,_,_,m_block), tQsQ, warp_id, lane_id, 
            seqlen_q - m_block * kBlockM, 
            /*clear_D=*/true);
        masked_copy<Is_even_MN>(
            gmem_tiled_copy_QKV, tdOgdO(_,_,_,m_block), tdOsdO, warp_id, lane_id, 
            seqlen_q - m_block * kBlockM, 
            /*clear_D=*/true);



        // copy(gmem_tiled_copy_QKV, tdOgdO(_,_,_,m_block), tdOrdO);

        // copy(gmem_tiled_copy_QKV, tQrQ, tQsQ);
        // copy(gmem_tiled_copy_QKV, tdOrdO, tdOsdO);




        __syncthreads();


        // somehow pipelining gmem loads for both Q and dO use alot more registers which is slower
//        if (m_block + 1 < m_block_no_mask) {
//            copy(gmem_tiled_copy_QKV, tQgQ(_,_,_,m_block+1), tQrQ);
//            //copy(gmem_tiled_copy_QKV, tdOgdO(_,_,_,q_tile+1), tdOrdO);
//        }

        // compute S=QK^T



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





        __syncthreads();

        // load rL, rD from gmem to rmem

        int global_row_offset = m_block * kBlockM;

        for (int i=0;i<2;i++) {
            for (int j=0;j<2;j++) {
                int block_row = warp_offset + thread_offset + 8 * j + 32 * i;
                int global_row = global_row_offset + block_row;
                if (global_row < seqlen_q) {
                    rL[i][j] = gL(block_row, m_block);
                    rD[i][j] = gD(block_row, m_block);
                }
            }
        }



        // rescale S
        for (int i=0;i< tSrS_float.size();i ++ ) {
            tSrS_float[i] *= softmax_scale;
        }



                // when Is_even_MN == false, we need to mask out the 'side' of the last n block
        // when Is_even_MN == false, we need to mask out the 'bottom' of the last m block
        // if (!Is_even_MN && m_block == m_block_max - 1) {
        if (!Is_even_MN && (n_block == n_block_max - 1 || m_block == m_block_max - 1)) {
            accum_SdP_mask.template apply_mask_bwd_dk_dv</*Is_causal=*/Is_causal, /*Is_even_MN=*/false>(
                tSrS_float, tdPrdP_float, warp_id, lane_id, m_block, n_block, seqlen_q, seqlen_k, kBlockM, kBlockN, head_dim
            );
        } else {
            accum_SdP_mask.template apply_mask_bwd_dk_dv</*Is_causal=*/Is_causal, /*Is_even_MN=*/true>(
                tSrS_float, tdPrdP_float, warp_id, lane_id, m_block, n_block, seqlen_q, seqlen_k, kBlockM, kBlockN, head_dim
        );

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



        Tensor tSrP = convert_type<half_t>(tSrP_float);


        Tensor tdPrdS = convert_type<half_t>(tdPrdS_float);


        // copy(gmem_tiled_copy_QKV, tVsV, tVrV);
        masked_copy<Is_even_MN>(
            gmem_tiled_copy_QKV, tVsV, tVrV, warp_id, lane_id, 
            seqlen_k - n_block * kBlockN, 
            /*clear_D=*/true);


        __syncthreads();

        copy(tSrP, tSsP);
        copy(tdPrdS, tdPsdS);


        __syncthreads();



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

    CUTE_NO_UNROLL

    for (; m_block < m_block_max; ++m_block) {
        copy(gmem_tiled_copy_QKV, tVrV, tVsV);


        clear(tSrS_float);
        clear(tdPrdP_float);

//        copy(gmem_tiled_copy_QKV, tQrQ, tQsQ);
//        copy(gmem_tiled_copy_QKV, tdOrdO, tdOsdO);

        // load gQ to sQ
    //    copy(gmem_tiled_copy_QKV, tQgQ(_,_,_,m_block), tQsQ);
    //    copy(gmem_tiled_copy_QKV, tdOgdO(_,_,_,m_block), tdOsdO);

         masked_copy<Is_even_MN>(
            gmem_tiled_copy_QKV, tQgQ(_,_,_,m_block), tQsQ, warp_id, lane_id, 
            seqlen_q - m_block * kBlockM, 
            /*clear_D=*/true);
         masked_copy<Is_even_MN>(
            gmem_tiled_copy_QKV, tdOgdO(_,_,_,m_block), tdOsdO, warp_id, lane_id, 
            seqlen_q - m_block * kBlockM, 
            /*clear_D=*/true);

        __syncthreads();





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





        __syncthreads();


        int global_row_offset = m_block * kBlockM;
        for (int i=0;i<2;i++) {
            for (int j=0;j<2;j++) {
                int block_row = warp_offset + thread_offset + 8 * j + 32 * i;
                int global_row = global_row_offset + block_row;

                if (global_row < seqlen_q) {
                    rL[i][j] = gL(block_row, m_block);
                    rD[i][j] = gD(block_row, m_block);
                }

            }
        }

        // rescale S
        for (int i=0;i< tSrS_float.size();i ++ ) {
            tSrS_float[i] *= softmax_scale;
        }


        // when Is_even_MN == false, we need to mask out the 'side' of the last n block
        // when Is_even_MN == false, we need to mask out the 'bottom' of the last m block
        // if (!Is_even_MN && m_block == m_block_max - 1) {
        if (!Is_even_MN && (n_block == n_block_max - 1 || m_block == m_block_max - 1)) {
            accum_SdP_mask.template apply_mask_bwd_dk_dv</*Is_causal=*/false, /*Is_even_MN=*/Is_even_MN>(
                tSrS_float, tdPrdP_float, warp_id, lane_id, m_block, n_block, seqlen_q, seqlen_k, kBlockM, kBlockN, head_dim
            );
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

        Tensor tSrP = convert_type<half_t>(tSrP_float);


        Tensor tdPrdS = convert_type<half_t>(tdPrdS_float);


        copy(gmem_tiled_copy_QKV, tVsV, tVrV);
        __syncthreads();

        copy(tSrP, tSsP);
        copy(tdPrdS, tdPsdS);

        __syncthreads();








        // dV += P^TdO
        CUTE_UNROLL
        for (int ptdot_block = 0; ptdot_block < PtdOt_BLOCK_MAX; ptdot_block++) {
            copy(smem_tiled_copy_dSt, tdVsPt_copy_view(_,_,ptdot_block), tdVrPt_copy_view(_,_,ptdot_block));
            copy(smem_tiled_copy_Qt, tdVsdOt_copy_view(_,_,ptdot_block), tdVrdOt_copy_view(_,_,ptdot_block));

 
            gemm(tiled_mma_dV, tdVrPt(_,_,ptdot_block), tdVrdOt(_,_,ptdot_block), tdVrdV_float);

        }



        // dK += dS^TQ

        CUTE_UNROLL
        for (int ptdot_block = 0; ptdot_block < PtdOt_BLOCK_MAX; ptdot_block++) {
            copy(smem_tiled_copy_dSt, tdKsdSt_copy_view(_,_,ptdot_block), tdKrdSt_copy_view(_,_,ptdot_block));
            copy(smem_tiled_copy_Qt, tdKsQt_copy_view(_,_,ptdot_block), tdKrQt_copy_view(_,_,ptdot_block));

            gemm(tiled_mma_dK, tdKrdSt(_,_,ptdot_block), tdKrQt(_,_,ptdot_block), tdKrdK_float);

        }



        __syncthreads();

    }
    


    // dV

    // dK

    // rescale by head dim
    for (int i=0;i< tdKrdK_float.size();i ++ ) {
        tdKrdK_float[i] *= softmax_scale;
    }




    Tensor tdKrdK = convert_type<half_t>(tdKrdK_float);
    Tensor tdVrdV = convert_type<half_t>(tdVrdV_float);


    copy(tdVrdV, tdVsdV);
    copy(tdKrdK, tdKsdK);

    __syncthreads();

    //store to gmem
    masked_copy<Is_even_MN>(
        gmem_tiled_copy_QKV, tdKsdK_copy, tdKgdK_copy, warp_id, lane_id, 
        seqlen_k - n_block * kBlockN, 
        /*clear_D=*/false);
    masked_copy<Is_even_MN>(
        gmem_tiled_copy_QKV, tdVsdV_copy, tdVgdV_copy, warp_id, lane_id, 
        seqlen_k - n_block * kBlockN, 
        /*clear_D=*/false);


}



template <typename Kernel_traits, bool Is_causal, bool Is_even_MN>
inline __device__ void compute_dq(
    half_t * __restrict__ q_ptr,
    half_t * __restrict__ k_ptr,
    half_t * __restrict__ v_ptr,
    float *__restrict__ l_ptr,
    float * __restrict__ d_ptr,
    half_t * __restrict__ do_ptr,
    half_t* __restrict__ dq_ptr,
    int * __restrict__ cu_seqlens_q,
    int * __restrict__ cu_seqlens_k,
    int batch_size, int seqlen_q, int seqlen_k, int num_heads, int num_heads_k, int h_h_k_ratio, int head_dim, float softmax_scale, int is_causal
) {

    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    compute_dq_1rowblock<Kernel_traits, Is_causal, Is_even_MN>(q_ptr,
                                                    k_ptr,
                                                    v_ptr,
                                                    l_ptr,
                                                    d_ptr,
                                                    do_ptr,
                                                    dq_ptr,
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
                                                    is_causal,
                                                    bidb,
                                                    bidh,
                                                    m_block);

}

template <typename Kernel_traits, bool Is_causal, bool Is_even_MN>
inline __device__ void compute_dk_dv(
    half_t * __restrict__ q_ptr,
    half_t * __restrict__ k_ptr,
    half_t * __restrict__ v_ptr,
    float * __restrict__ l_ptr,
    float * __restrict__ d_ptr,
    half_t * __restrict__ do_ptr,
    half_t* __restrict__ dk_ptr,
    half_t* __restrict__ dv_ptr,
    int * __restrict__ cu_seqlens_q,
    int * __restrict__ cu_seqlens_k,
    int batch_size, int seqlen_q, int seqlen_k, int num_heads, int num_heads_k, int h_h_k_ratio, int head_dim, float softmax_scale, int is_causal
) {
    const int n_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    compute_dk_dv_1colblock<Kernel_traits, Is_causal, Is_even_MN>(q_ptr,
                                                    k_ptr,
                                                    v_ptr,
                                                    l_ptr,
                                                    d_ptr,
                                                    do_ptr,
                                                    dk_ptr,
                                                    dv_ptr,
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
                                                    is_causal,
                                                    bidb,
                                                    bidh,
                                                    n_block);

}
