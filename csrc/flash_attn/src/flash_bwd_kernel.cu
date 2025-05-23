#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <float.h>
#include <torch/extension.h>
#include <cute/tensor.hpp>
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include <cuda_fp16.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "kernel_traits.h"

using namespace cute;

// kBlockN = 64 works
// kBlockM = 64 doesnt work
#define K_BLOCK_M 64
#define K_BLOCK_N 64


__global__ __launch_bounds__(1024)
void compute_dot_do_o(half_t* o_ptr,
                      half_t* do_ptr,
                      float*  d_ptr,
                      int batch_size, int seq_len, int num_heads, int head_dim)
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


    int thread_row = warp_id;
    int thread_col = lane_id * 4;

    for (int i=0;i<4;i++) {
        rO[i] = o_ptr[do_o_offset + thread_row * num_heads * head_dim + thread_col + i];
        rdO[i] = do_ptr[do_o_offset + thread_row * num_heads * head_dim + thread_col + i];
    }

    // thread reduction
    for (int i=0;i<4;i ++) {
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

template <typename Kernel_traits>
__global__ __launch_bounds__(256)
void compute_dq_kernel(
    half_t const* __restrict__ q_ptr,
    half_t const* __restrict__ k_ptr,
    half_t const* __restrict__ v_ptr,
    float const* __restrict__ l_ptr,
    float const* __restrict__ d_ptr,
    half_t const* __restrict__ do_ptr,
    half_t* __restrict__ dq_ptr,
    int batch_size, int seq_len, int num_heads, int head_dim
)
{
    constexpr int kBlockM = K_BLOCK_M;
    constexpr int kBlockN = K_BLOCK_N;
    constexpr int kHeadDim = 128;
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

    Tensor gQ = local_tile(mQ(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(blockIdx.z, 0));


    // K
    Tensor mK = make_tensor(make_gmem_ptr(k_ptr),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gK = local_tile(mK(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));

    // V
    Tensor mV = make_tensor(make_gmem_ptr(v_ptr),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gV = local_tile(mV(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));


    // L = m + log l
    Tensor mL = make_tensor(make_gmem_ptr(l_ptr),
                             make_shape(batch_size, num_heads, seq_len),
                             make_stride(seq_len * num_heads,  seq_len, Int<1>{}));

    Tensor gL = local_tile(mL(blockIdx.x, blockIdx.y, _), Shape<Int<kBlockM>>{},
                           make_coord(blockIdx.z));


    Tensor mD = make_tensor(make_gmem_ptr(d_ptr),
                             make_shape(batch_size, num_heads, seq_len),
                             make_stride(seq_len * num_heads,  seq_len, Int<1>{}));

    Tensor gD = local_tile(mD(blockIdx.x, blockIdx.y, _), Shape<Int<kBlockM>>{},
                           make_coord(blockIdx.z));

    // dO
    Tensor mdO = make_tensor(make_gmem_ptr(do_ptr),
                             make_shape(batch_size, seq_len, num_heads, head_dim),
                             make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gdO = local_tile(mdO(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(blockIdx.z, 0));


    // dQ
    Tensor mdQ = make_tensor(make_gmem_ptr(dq_ptr),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gdQ = local_tile(mdQ(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(blockIdx.z, 0));


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

    auto KV_TILE_MAX = size<3>(tSgK);
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

        // convert dS from fp32 to fp16
        constexpr int num_element_dS = decltype(size(tdPrdS_float))::value;

        cutlass::NumericArrayConverter<half_t, float, num_element_dS> convert_op_dS;
        auto frag_dS = convert_op_dS(*reinterpret_cast<const cutlass::Array<float, num_element_dS> *>(tdPrdS_float.data()));

        Tensor tdPrdS = make_tensor(make_rmem_ptr<half_t>(&frag_dS), tdPrdS_float.layout());

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
        tdQrdQ_float[i] *= 1.0f / sqrtf(kHeadDim);
    }




    // dQ
    constexpr int num_element = decltype(size(tdQrdQ_float))::value;

    cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tdQrdQ_float.data()));

    Tensor tdQrdQ = make_tensor(make_rmem_ptr<half_t>(&frag), tdQrdQ_float.layout());

    copy(tdQrdQ, tdQsdQ);

    __syncthreads();

    copy(gmem_tiled_copy_QKV, tdQsdQ_copy, tdQgdQ_copy);

}



template <typename Kernel_traits>
__global__ __launch_bounds__(256)
void compute_dk_dv_kernel(
    half_t const* __restrict__ q_ptr,
    half_t const* __restrict__ k_ptr,
    half_t const* __restrict__ v_ptr,
    float const* __restrict__ l_ptr,
    float const* __restrict__ d_ptr,
    half_t const* __restrict__ do_ptr,
    half_t* __restrict__ dk_ptr,
    half_t* __restrict__ dv_ptr,
    int batch_size, int seq_len, int num_heads, int head_dim
)
{   
    constexpr int kBlockM = K_BLOCK_M;
    constexpr int kBlockN = K_BLOCK_N;
    constexpr int kHeadDim = 128;
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

    Tensor gQ = local_tile(mQ(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(_, 0));


    // K
    Tensor mK = make_tensor(make_gmem_ptr(k_ptr),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gK = local_tile(mK(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(blockIdx.z, 0));

    // V
    Tensor mV = make_tensor(make_gmem_ptr(v_ptr),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gV = local_tile(mV(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(blockIdx.z, 0));


    // L = m + log l
    Tensor mL = make_tensor(make_gmem_ptr(l_ptr),
                             make_shape(batch_size, num_heads, seq_len),
                             make_stride(seq_len * num_heads,  seq_len, Int<1>{}));

    Tensor gL = local_tile(mL(blockIdx.x, blockIdx.y, _), Shape<Int<kBlockM>>{},
                           make_coord(_));


    Tensor mD = make_tensor(make_gmem_ptr(d_ptr),
                             make_shape(batch_size, num_heads, seq_len),
                             make_stride(seq_len * num_heads,  seq_len, Int<1>{}));

    Tensor gD = local_tile(mD(blockIdx.x, blockIdx.y, _), Shape<Int<kBlockM>>{},
                           make_coord(_));

    // dO
    Tensor mdO = make_tensor(make_gmem_ptr(do_ptr),
                             make_shape(batch_size, seq_len, num_heads, head_dim),
                             make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gdO = local_tile(mdO(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(_, 0));

    // dV
    Tensor mdV = make_tensor(make_gmem_ptr(dv_ptr),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gdV = local_tile(mdV(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(blockIdx.z, 0));

    // dK
    Tensor mdK = make_tensor(make_gmem_ptr(dk_ptr),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gdK = local_tile(mdK(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(blockIdx.z, 0));



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

    auto smem_tiled_copy_Pt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQtdOtPtdSt{}, tiled_mma_dV);
    auto smem_thr_copy_Pt = smem_tiled_copy_Pt.get_slice(threadIdx.x);
    auto tdVsPt_copy_view = smem_thr_copy_Pt.partition_S(sPt);
    auto tdVrPt_copy_view = smem_thr_copy_Pt.retile_D(tdVrPt);

    auto smem_tiled_copy_dOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomQtdOtPtdSt{}, tiled_mma_dV);
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

    auto smem_tiled_copy_dSt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomQtdOtPtdSt{}, tiled_mma_dK);
    auto smem_thr_copy_dSt = smem_tiled_copy_dSt.get_slice(threadIdx.x);
    auto tdKsdSt_copy_view = smem_thr_copy_dSt.partition_S(sdSt);
    auto tdKrdSt_copy_view = smem_thr_copy_dSt.retile_D(tdKrdSt);

    auto smem_tiled_copy_Qt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomQtdOtPtdSt{}, tiled_mma_dK);
    auto smem_thr_copy_Qt = smem_tiled_copy_Qt.get_slice(threadIdx.x);
    auto tdKsQt_copy_view = smem_thr_copy_dOt.partition_S(sQt);
    auto tdKrQt_copy_view = smem_thr_copy_dOt.retile_D(tdKrQt);



    auto Q_TILE_MAX = size<3>(tSgQ);
    auto QK_BLOCK_MAX = size<2>(tSsK);
    auto PtdOt_BLOCK_MAX = size<2>(tdVsPt);
    // load K, V, dK, dV tiles

    copy(gmem_tiled_copy_QKV, tKgK, tKsK);
    //copy(tSgK, tSsK);
    copy(gmem_tiled_copy_QKV, tVgV, tVrV);


    clear(tdVrdV_float);
    clear(tdKrdK_float);


    copy(gmem_tiled_copy_QKV, tQgQ(_,_,_,0), tQrQ);
    //copy(gmem_tiled_copy_QKV, tdOgdO(_,_,_,0), tdOrdO);

    CUTE_NO_UNROLL
    for (int q_tile = 0; q_tile < Q_TILE_MAX; ++q_tile) {
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
        constexpr int num_element = decltype(size(tSrP_float))::value;

        cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
        auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tSrP_float.data()));

        Tensor tSrP = make_tensor(make_rmem_ptr<half_t>(&frag), tSrP_float.layout());
//
//
        // convert dS from fp32 to fp16
        constexpr int num_element_dS = decltype(size(tdPrdS_float))::value;

        cutlass::NumericArrayConverter<half_t, float, num_element_dS> convert_op_dS;
        auto frag_dS = convert_op_dS(*reinterpret_cast<const cutlass::Array<float, num_element_dS> *>(tdPrdS_float.data()));

        Tensor tdPrdS = make_tensor(make_rmem_ptr<half_t>(&frag_dS), tdPrdS_float.layout());
//
//
        copy(gmem_tiled_copy_QKV, tVsV, tVrV);
        __syncthreads();

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

    // dV
    constexpr int num_element = decltype(size(tdVrdV_float))::value;

    cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tdVrdV_float.data()));

    Tensor tdVrdV = make_tensor(make_rmem_ptr<half_t>(&frag), tdVrdV_float.layout());

    copy(tdVrdV, tdVsdV);
//
//
    // dK

    // rescale by head dim
    for (int i=0;i< tdKrdK_float.size();i ++ ) {
        tdKrdK_float[i] *= 1.0f / sqrtf(kHeadDim);
    }


    constexpr int num_element_dK = decltype(size(tdKrdK_float))::value;

    cutlass::NumericArrayConverter<half_t, float, num_element_dK> convert_op_dK;
    auto frag_dK = convert_op_dK(*reinterpret_cast<const cutlass::Array<float, num_element_dK> *>(tdKrdK_float.data()));

    Tensor tdKrdK = make_tensor(make_rmem_ptr<half_t>(&frag_dK), tdKrdK_float.layout());

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





std::vector<torch::Tensor>
flash_bwd(torch::Tensor q,
          torch::Tensor k,
          torch::Tensor v,
          torch::Tensor o,
          torch::Tensor l,
          torch::Tensor d_o,
          int batch_size, int seq_len, int num_heads, int head_dim)
{

    constexpr int kBlockM = K_BLOCK_M;
    constexpr int kBlockN = K_BLOCK_N;
    constexpr int kHeadDim = 128;

    torch::Tensor dq = torch::zeros(q.sizes(), q.options().dtype(torch::kFloat16));
    torch::Tensor dk = torch::empty(k.sizes(), k.options().dtype(torch::kFloat16));
    torch::Tensor dv = torch::empty(v.sizes(), v.options().dtype(torch::kFloat16));
    torch::Tensor d = torch::empty(l.sizes(), l.options());

    half_t* q_ptr = reinterpret_cast<half_t*>(q.data_ptr());
    half_t* k_ptr = reinterpret_cast<half_t*>(k.data_ptr());
    half_t* v_ptr = reinterpret_cast<half_t*>(v.data_ptr());
    half_t* o_ptr = reinterpret_cast<half_t*>(o.data_ptr());
    float* l_ptr = reinterpret_cast<float*>(l.data_ptr());
    float* d_ptr = reinterpret_cast<float*>(d.data_ptr());
    half_t* do_ptr = reinterpret_cast<half_t*>(d_o.data_ptr());

    half_t* dq_ptr = reinterpret_cast<half_t*>(dq.data_ptr());
    half_t* dk_ptr = reinterpret_cast<half_t*>(dk.data_ptr());
    half_t* dv_ptr = reinterpret_cast<half_t*>(dv.data_ptr());

//     for (int i=0;i<1024;i++) {
//         printf("check o, i = %d, o = %f", i, static_cast<float>(o_ptr[i]));
//     }

    // compute dO \circ O
    // we use 1 warp to compute a single row
    // each thread block we launch 1024 = 32 x 32 threads = 32 warps
    // so each thread block process 32 rows
    dim3 dimGrid_dot_do_o(batch_size, num_heads, seq_len / 32);
    dim3 dimBlock_dot_do_o(1024);
    compute_dot_do_o<<<dimGrid_dot_do_o, dimBlock_dot_do_o>>>(o_ptr,
                    do_ptr,
                    d_ptr,
                    batch_size, seq_len, num_heads, head_dim);

    int maxbytes = 65536;




    // compute dQ
    dim3 dimGrid_dq(batch_size, num_heads, seq_len / kBlockM);
    dim3 dimBlock_dq(256);

    auto dq_kernel = compute_dq_kernel<Flash_bwd_kernel_traits<kHeadDim, kBlockM, kBlockN, 8>>;
    cudaFuncSetAttribute(dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    dq_kernel<<<dimGrid_dq, dimBlock_dq, maxbytes>>>(q_ptr,
                                            k_ptr,
                                            v_ptr,
                                            l_ptr,
                                            d_ptr,
                                            do_ptr,
                                            dq_ptr,
                                            batch_size, seq_len, num_heads, head_dim);


    // compute dK, dV
    dim3 dimGrid(batch_size, num_heads, seq_len / kBlockN);
    dim3 dimBlock(256);

    auto dk_dv_kernel = compute_dk_dv_kernel<Flash_bwd_kernel_traits<kHeadDim, kBlockM, kBlockN, 8>>;
    cudaFuncSetAttribute(dk_dv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    dk_dv_kernel<<<dimGrid, dimBlock, maxbytes>>>(q_ptr,
                                            k_ptr,
                                            v_ptr,
                                            l_ptr,
                                            d_ptr,
                                            do_ptr,
                                            dk_ptr,
                                            dv_ptr,
                                            batch_size, seq_len, num_heads, head_dim);


    return { dq, dk, dv };

}