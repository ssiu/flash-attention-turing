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

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

using namespace cute;


#define HEAD_SIZE 128
#define Q_TILE_SIZE 128
#define KV_TILE_SIZE 64



template <class SmemLayoutQ, class TiledCopyQ, class TiledMmaS,
          class SmemLayoutK, class TiledCopyK, class TiledMmaO,
          class SmemLayoutV, class TiledCopyV,
          class SmemLayoutS,
          class SmemLayoutO, class TiledCopyO>
__global__ __launch_bounds__(256)
void flash_fwd_kernel(
    half_t const* q, SmemLayoutQ sQ_layout, TiledCopyQ copy_Q, TiledMmaS mma_S,
    half_t const* k, SmemLayoutK sK_layout, TiledCopyK copy_K, TiledMmaO mma_O,
    half_t const* v, SmemLayoutV sV_layout, TiledCopyV copy_V,
                     SmemLayoutS sS_layout,
    half_t* o,       SmemLayoutO sO_layout, TiledCopyO copy_O,
    int batch_size, int seq_len, int num_heads, int head_dim
)
{


    Tensor mQ = make_tensor(make_gmem_ptr(q),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gQ = local_tile(mQ(blockIdx.x, _, blockIdx.y, _), Shape<Int<Q_TILE_SIZE>, Int<HEAD_SIZE>>{},
                           make_coord(blockIdx.z, 0));

    Tensor mK = make_tensor(make_gmem_ptr(k),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gK = local_tile(mK(blockIdx.x, _, blockIdx.y, _), Shape<Int<KV_TILE_SIZE>, Int<HEAD_SIZE>>{},
                           make_coord(_, 0));

    // this is a (seq_len, head_dim) column major matrix, so its V^T in row major
    Tensor mV = make_tensor(make_gmem_ptr(v),
                            make_shape(batch_size, head_dim, num_heads, seq_len),
                            make_stride(seq_len * num_heads * head_dim, Int<1>{}, head_dim, num_heads * head_dim));

    Tensor gV = local_tile(mV(blockIdx.x, _, blockIdx.y, _), Shape<Int<HEAD_SIZE>, Int<KV_TILE_SIZE>>{},
                           make_coord(0, _));

    Tensor mO = make_tensor(make_gmem_ptr(o),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gO = local_tile(mO(blockIdx.x, _, blockIdx.y, _), Shape<Int<Q_TILE_SIZE>, Int<HEAD_SIZE>>{},
                           make_coord(blockIdx.z, 0));


    extern __shared__ char smem_[];


    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), sQ_layout);
    Tensor sK = make_tensor(sQ.data() + Q_TILE_SIZE*HEAD_SIZE, sK_layout);
    Tensor sV = make_tensor(sK.data(), sV_layout);
    Tensor sP = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), sS_layout);

    Tensor sS = make_tensor(make_smem_ptr(reinterpret_cast<float*>(&smem_[0])), sS_layout);
    Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), sO_layout);
    Tensor sO_float = make_tensor(make_smem_ptr(reinterpret_cast<float*>(&smem_[0])), sO_layout);


    int thread_id = threadIdx.x;
    int lane_id = thread_id % 32;


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
    ThrCopy thr_copy_Q = copy_Q.get_slice(threadIdx.x);
    Tensor tQgQ = thr_copy_Q.partition_S(gQ);
    Tensor tQsQ = thr_copy_Q.partition_D(sQ);


    ThrCopy thr_copy_K = copy_K.get_slice(threadIdx.x);
    Tensor tKgK = thr_copy_K.partition_S(gK);
    Tensor tKsK = thr_copy_K.partition_D(sK);
    Tensor tKrK = make_fragment_like(tKsK);

    ThrCopy thr_copy_V = copy_V.get_slice(threadIdx.x);
    Tensor tVgV = thr_copy_V.partition_S(gV);
    Tensor tVsV = thr_copy_V.partition_D(sV);
    Tensor tVrV = make_fragment_like(tVsV);

    // smem -> gmem for O
    ThrCopy thr_copy_O = copy_O.get_slice(threadIdx.x);
    Tensor tOsO_copy = thr_copy_O.partition_S(sO);
    Tensor tOgO_copy = thr_copy_O.partition_D(gO);

    // mma for S = QK^T
    ThrMMA thr_mma_S = mma_S.get_slice(threadIdx.x);
    Tensor tSsQ = thr_mma_S.partition_A(sQ);
    Tensor tSsK = thr_mma_S.partition_B(sK);
    Tensor tSsS = thr_mma_S.partition_C(sS);

    Tensor tSrQ = thr_mma_S.make_fragment_A(tSsQ);
    Tensor tSrK = thr_mma_S.make_fragment_B(tSsK);
    Tensor tSrS = thr_mma_S.make_fragment_C(tSsS);


    // mma for O = PV
    ThrMMA thr_mma_O = mma_O.get_slice(threadIdx.x);
    Tensor tOsP = thr_mma_O.partition_A(sP);
    Tensor tOsV = thr_mma_O.partition_B(sV);

    Tensor tOrP = thr_mma_O.make_fragment_A(tOsP);
    Tensor tOrV = thr_mma_O.make_fragment_B(tOsV);

    // tOsO_float is not used, only used to construct the register tensor
    Tensor tOsO_float = thr_mma_O.partition_C(sO_float);
    Tensor tOrO_float = thr_mma_O.make_fragment_C(tOsO_float);

    Tensor tOsO = thr_mma_O.partition_C(sO);
    //  each warp only process 16 rows
    auto s2r_tiled_copy_Q = make_tiled_copy_A(Copy_Atom<SM75_U32x2_LDSM_N, half_t>{}, mma_S);
    auto s2r_thr_copy_Q = s2r_tiled_copy_Q.get_slice(threadIdx.x);
    auto tSsQ_copy_view = s2r_thr_copy_Q.partition_S(sQ);
    auto tSrQ_copy_view = s2r_thr_copy_Q.retile_D(tSrQ);

    auto s2r_tiled_copy_K = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, mma_S);
    auto s2r_thr_copy_K = s2r_tiled_copy_K.get_slice(threadIdx.x);
    auto tSsK_copy_view = s2r_thr_copy_K.partition_S(sK);
    auto tSrK_copy_view = s2r_thr_copy_K.retile_D(tSrK);

    auto s2r_tiled_copy_V = make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, half_t>{}, mma_O);
    auto s2r_thr_copy_V = s2r_tiled_copy_V.get_slice(threadIdx.x);
    auto tOsV_copy_view = s2r_thr_copy_V.partition_S(sV);
    auto tOrV_copy_view = s2r_thr_copy_V.retile_D(tOrV);

    auto KV_TILE_MAX = size<3>(tKgK);
    auto QK_BLOCK_MAX = size<2>(tSsK);
    auto PV_BLOCK_MAX = size<2>(tOsV);
    // prologue

    copy(copy_Q, tQgQ, tQsQ);
    copy(copy_K, tKgK(_,_,_,0), tKrK);


    clear(tOrO_float);

    // main loop
    CUTE_NO_UNROLL
    for (int kv_tile = 0; kv_tile < KV_TILE_MAX; ++kv_tile) {


        copy(copy_K, tKrK, tKsK);

        __syncthreads();

        clear(tSrS);

        if (kv_tile + 1 < KV_TILE_MAX) {
            copy(copy_K, tKgK(_,_,_,kv_tile + 1), tKrK);
        }


        for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
            copy(s2r_tiled_copy_Q, tSsQ_copy_view(_,_,qk_block), tSrQ_copy_view(_,_,qk_block));
            copy(s2r_tiled_copy_K, tSsK_copy_view(_,_,qk_block), tSrK_copy_view(_,_,qk_block));

            gemm(mma_S, tSrQ(_,_,qk_block), tSrK(_,_,qk_block), tSrS);

        }

        __syncthreads();
        copy(copy_V, tVgV(_,_,_,kv_tile), tVsV);
        __syncthreads();

        for (int i=0;i< tSrS.size();i ++ ) {
            tSrS[i] *= 1.0f / sqrtf(HEAD_SIZE);
        }


        // compute m = rowmax(S)
        for (int i=0; i< 2; i++) {
            rM[i] = rM_old[i];
        }


        // intra-thread reduction

        for (int i=0; i< 2; i++) {
            for (int j=0; j < tSrS(make_coord(_,i),_,_).size(); j++) {
                rM[i] = fmaxf(rM[i], tSrS(make_coord(_,i),_,_)[j]);
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
            for (int j=0; j < tSrS(make_coord(_,i),_,_).size(); j++) {
                tSrS(make_coord(_,i),_,_)[j] = expf(tSrS(make_coord(_,i),_,_)[j] - rM[i]);
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
            for (int j=0; j < tSrS(make_coord(_,i),_,_).size(); j++) {
                rD[i] += tSrS(make_coord(_,i),_,_)[j];
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

        // sync rL
        for (int i =0; i<2; i++) {
            rL[i] = __shfl_sync(mask, rL[i], lane_id_to_read_from);
        }



        constexpr int num_element = decltype(size(tSrS))::value;

        cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
        auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tSrS.data()));

        Tensor tOrP = make_tensor(make_rmem_ptr<half_t>(&frag), tSrS.layout());



        // rescale O

        for (int i =0; i<2; i++) {
            for (int j=0; j < tOrO_float(make_coord(_,i),_,_).size(); j++) {
                tOrO_float(make_coord(_,i),_,_)[j] = expf(rM_old[i] - rM[i]) * tOrO_float(make_coord(_,i),_,_)[j];
            }
        }




        for (int pv_block = 0; pv_block < PV_BLOCK_MAX; pv_block++) {


            copy(s2r_tiled_copy_V, tOsV_copy_view(_,_,pv_block), tOrV_copy_view(_,_,pv_block));

            gemm(mma_O, tOrP(_,_,pv_block), tOrV(_,_,pv_block), tOrO_float);

        }


        // update m and l
        for (int i = 0; i< 2;i++) {
            rM_old[i] = rM[i];
            rL_old[i] = rL[i];
        }


        __syncthreads();

    }
    // end of KV loop


    for (int i =0; i<2; i++) {
        for (int j=0; j < tOrO_float(make_coord(_,i),_,_).size(); j++) {
            tOrO_float(make_coord(_,i),_,_)[j] /= rL[i];
        }
    }


    constexpr int num_element = decltype(size(tOrO_float))::value;

    cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tOrO_float.data()));
    Tensor tOrO = make_tensor(make_rmem_ptr<half_t>(&frag), tOrO_float.layout());


    copy(tOrO, tOsO);

    __syncthreads();

    copy(copy_O, tOsO_copy, tOgO_copy);

}



torch::Tensor flash_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                int batch_size, int seq_len, int num_heads, int head_dim)
{


    auto sQ_layout_atom = composition(Swizzle<3, 3, 3>{},
                                 Layout<Shape<_16,_64>,
                                 Stride<_64, _1>>{});

    auto sK_layout_atom = composition(Swizzle<3, 3, 3>{},
                               Layout<Shape<_16,_64>,
                               Stride<_64, _1>>{});

    auto sV_layout_atom = composition(Swizzle<3, 3, 3>{},
                             Layout<Shape<_64,_16>,
                             Stride<_1, _64>>{});

    auto sO_layout_atom = composition(Swizzle<3, 3, 3>{},
                                 Layout<Shape<_16,_64>,
                                 Stride<_64, _1>>{});


    auto sQ_layout = tile_to_shape(sQ_layout_atom,
                           make_shape(Int<Q_TILE_SIZE>{}, Int<HEAD_SIZE>{}));



    auto sK_layout = tile_to_shape(sK_layout_atom,
                           make_shape(Int<KV_TILE_SIZE>{}, Int<HEAD_SIZE>{}));


    auto sV_layout = tile_to_shape(sV_layout_atom,
                            make_shape(Int<HEAD_SIZE>{}, Int<KV_TILE_SIZE>{}));

    auto sO_layout = tile_to_shape(sO_layout_atom,
                           make_shape(Int<Q_TILE_SIZE>{}, Int<HEAD_SIZE>{}));



    auto sS_layout = make_layout(make_shape (Int<Q_TILE_SIZE>{}, Int<KV_TILE_SIZE>{}),
                        make_stride(Int<KV_TILE_SIZE>{}, Int<1>{}));





    TiledCopy copy_Q = make_tiled_copy(Copy_Atom<AutoVectorizingCopy, half_t>{},
                                Layout<Shape<_32,_8>, Stride<_8,_1>>{},
                                Layout<Shape< _1,_8>>{});

    TiledCopy copy_K = make_tiled_copy(Copy_Atom<AutoVectorizingCopy, half_t>{},
                                Layout<Shape<_32,_8>, Stride<_8,_1>>{},
                                Layout<Shape< _1,_8>>{});

    // 64 threads loading a 128 x 32 tile
    TiledCopy copy_V = make_tiled_copy(Copy_Atom<AutoVectorizingCopy, half_t>{},
                                Layout<Shape<_8,_32>, Stride<_1,_8>>{},
                                Layout<Shape< _8,_1>>{});

    TiledCopy copy_O = make_tiled_copy(Copy_Atom<AutoVectorizingCopy, half_t>{},
                                Layout<Shape<_32,_8>, Stride<_8,_1>>{},
                                Layout<Shape< _1,_8>>{});


    TiledMMA mma_S = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                                        Layout<Shape<_8, _1, _1>>{},
                                        Tile<_128,_16,_16>{}); // (Q_TILE_SIZE, KV_TILE_SIZE, 8)

    TiledMMA mma_O = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                                        Layout<Shape<_8, _1, _1>>{},
                                        Tile<_128,_16,_16>{}); // (Q_TILE_SIZE, HEAD_SIZE, 8)


    torch::Tensor o = torch::empty(q.sizes(), q.options().dtype(torch::kFloat16));

    half_t* q_ptr = reinterpret_cast<half_t*>(q.data_ptr());
    half_t* k_ptr = reinterpret_cast<half_t*>(k.data_ptr());
    half_t* v_ptr = reinterpret_cast<half_t*>(v.data_ptr());
    half_t* o_ptr = reinterpret_cast<half_t*>(o.data_ptr());


    dim3 dimGrid(batch_size, num_heads, seq_len / Q_TILE_SIZE);
    dim3 dimBlock(256);
    int maxbytes = 65536;


    auto kernel = flash_fwd_kernel<decltype(sQ_layout), decltype(copy_Q), decltype(mma_S),
                                      decltype(sK_layout), decltype(copy_K), decltype(mma_O),
                                      decltype(sV_layout), decltype(copy_V),
                                      decltype(sS_layout),
                                      decltype(sO_layout), decltype(copy_O)>;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    flash_fwd_kernel<<<dimGrid, dimBlock, maxbytes>>>(q_ptr, sQ_layout, copy_Q, mma_S,
                                                         k_ptr, sK_layout, copy_K, mma_O,
                                                         v_ptr, sV_layout, copy_V,
                                                                sS_layout,
                                                         o_ptr, sO_layout, copy_O,
                                                         batch_size, seq_len, num_heads, head_dim);

    return o;

}