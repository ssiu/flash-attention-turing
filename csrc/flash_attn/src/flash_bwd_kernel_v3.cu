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
#define K_BLOCK_N 32


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

    int thread_id = threadIdx.x;
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



__global__ __launch_bounds__(256)
void compute_dq_dk_dv_kernel_v3(
    half_t const* q_ptr,
    half_t const* k_ptr,
    half_t const* v_ptr,
    float const* l_ptr,
    float const* d_ptr,
    half_t const* do_ptr,
    half_t* dk_ptr,
    half_t* dv_ptr,
    half_t* dq_ptr,
    int batch_size, int seq_len, int num_heads, int head_dim
)
{   
    constexpr int kBlockM = K_BLOCK_M;
    constexpr int kBlockN = K_BLOCK_N;
    constexpr int kHeadDim = 128;
    constexpr int kNWarps = 8;
    constexpr int kNThreads = kNWarps * 32;

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

    using TiledMma_S = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_2, Int<kNWarps/2>, _1>>,
        Tile<Int<kBlockM>, Int<kBlockN>, _8>>;

    using TiledMma_dP = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_2, Int<kNWarps/2>, _1>>,
        Tile<Int<kBlockM>, Int<kBlockN>, _8>>;

    using TiledMma_dV = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_2, Int<kNWarps/2>, _1>>,
        Tile<Int<kBlockN>, Int<kHeadDim>, _8>>;

    using TiledMma_dK = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_2, Int<kNWarps/2>, _1>>,
        Tile<Int<kBlockN>, Int<kHeadDim>, _8>>;


    using TiledMma_dQ = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_2, Int<kNWarps/2>, _1>>,
        Tile<Int<kBlockM>, Int<kHeadDim>, _8>>;


    using Gmem_copy_struct = AutoVectorizingCopyWithAssumedAlignment<128>;

    using GmemLayoutAtomQKV = Layout<Shape <Int<kNThreads / 8>, _8>, Stride<_8, _1>>;

    using GmemTiledCopyQKV = decltype(
                make_tiled_copy(Copy_Atom<Gmem_copy_struct, half_t>{},
                                GmemLayoutAtomQKV{},
                                Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read


    using SmemLayoutAtom = decltype(
                    Layout<Shape<Int<kBlockM>, Int<kBlockN>>,
                    Stride<Int<kBlockN>, _1>>{});

    using SmemLayoutAtomTranposed = decltype(
                    Layout<Shape<Int<kBlockN>, Int<kBlockM>>,
                    Stride<_1, Int<kBlockN>>>{});
    
    using SmemLayoutQ = decltype(
                            Layout<Shape<Int<kBlockM>, Int<kHeadDim>>,
                            Stride<Int<kHeadDim>, _1>>{});

    using SmemLayoutQTransposed = decltype(
                                      Layout<Shape<Int<kHeadDim>, Int<kBlockM>>,
                                      Stride<_1, Int<kHeadDim>>>{});



    using SmemLayoutKV = decltype(
           Layout<Shape<Int<kBlockN>, Int<kHeadDim>>,
           Stride<Int<kHeadDim>, _1>>{});

    using SmemLayoutKVTransposed = decltype(
           Layout<Shape<Int<kHeadDim>, Int<kBlockN>>,
           Stride<_1, Int<kHeadDim>>>{});


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
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), SmemLayoutQ{});
    Tensor sQt = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), SmemLayoutQTransposed{});
    //Tensor sK = make_tensor(sQ.data() + kBlockM * kHeadDim, SmemLayoutKV{});

    // 64 * 128 = 16KB
    Tensor sK = make_tensor(sQ.data() + size(sQ), SmemLayoutKV{});
    Tensor sKt = make_tensor(sQ.data() + size(sQ), SmemLayoutKVTransposed{});

//     // 64 * 128 = 16KB
//     Tensor sdO = make_tensor(sK.data() + size(sK), SmemLayoutQ{});
//     Tensor sdOt = make_tensor(sK.data() + size(sK), SmemLayoutQTransposed{});
//
//
//     // 64 * 128 = 16KB
//     Tensor sV = make_tensor(sdO.data() + size(sdO), SmemLayoutKV{});
//
//
//     // 64 * 64 = 8KB
//     Tensor sP = make_tensor(sdO.data() + size(sdO), SmemLayoutAtom{});
//     Tensor sPt = make_tensor(sdO.data() + size(sdO), SmemLayoutAtomTranposed{});
//
//     // 64 * 64 = 8KB
//     Tensor sdS = make_tensor(sP.data() + size(sP), SmemLayoutAtom{});
//     Tensor sdSt = make_tensor(sP.data() + size(sP), SmemLayoutAtomTranposed{});



    int thread_id = threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    //int thread_row = warp_id * 16 + lane_id / 4;
    int warp_offset = (warp_id % 2) * 16;
    int thread_offset = lane_id / 4;

    float rL[2][2] = {0};
    float rD[2][2] = {0};

    // Copy operation
    GmemTiledCopyQKV gmem_tiled_copy_QKV;

    ThrCopy thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(threadIdx.x);

    Tensor tQgQ = thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = thr_copy_QKV.partition_D(sQ);

    Tensor tKgK = thr_copy_QKV.partition_S(gK);
    Tensor tKsK = thr_copy_QKV.partition_D(sK);

//     Tensor tVgV = thr_copy_QKV.partition_S(gV);
//     Tensor tVsV = thr_copy_QKV.partition_D(sV);
//
//     Tensor tdOgdO = thr_copy_QKV.partition_S(gdO);
//     Tensor tdOsdO = thr_copy_QKV.partition_D(sdO);


    // S = QK^T
    TiledMma_S tiled_mma_S;
    ThrMMA thr_mma_S = tiled_mma_S.get_slice(threadIdx.x);
    Tensor tSgQ = thr_mma_S.partition_A(gQ);
    Tensor tSsQ = thr_mma_S.partition_A(sQ);
    Tensor tSrQ = thr_mma_S.make_fragment_A(tSsQ);

    Tensor tSgK = thr_mma_S.partition_B(gK);
    Tensor tSsK = thr_mma_S.partition_B(sK);
    Tensor tSrK = thr_mma_S.make_fragment_B(tSsK);

    Tensor tSrS_float = partition_fragment_C(tiled_mma_S, Shape<Int<kBlockM>, Int<kBlockN>>{});
//     Tensor tSsP = thr_mma_S.partition_C(sP);
    //Tensor tSsS_float = thr_mma_S.partition_C(sS);


//     // dP = dOV^T
//     TiledMma_dP tiled_mma_dP;
//     ThrMMA thr_mma_dP = tiled_mma_dP.get_slice(threadIdx.x);
//     Tensor tdPgdO = thr_mma_dP.partition_A(gdO);
//     Tensor tdPsdO = thr_mma_dP.partition_A(sdO);
//     Tensor tdPgV = thr_mma_dP.partition_B(gV);
//     Tensor tdPsV = thr_mma_dP.partition_B(sV);
//     Tensor tdPrdP_float = partition_fragment_C(tiled_mma_dP, Shape<Int<kBlockM>, Int<kBlockN>>{});
//     Tensor tdPsdS = thr_mma_dP.partition_C(sdS);
//
//
//     // dV += P^TdO
//     TiledMma_dV tiled_mma_dV;
//     ThrMMA thr_mma_dV = tiled_mma_dV.get_slice(threadIdx.x);
//     Tensor tdVsPt = thr_mma_dV.partition_A(sPt);
//     // for copying dO from gmem to smem
//     Tensor tdVgdO = thr_mma_dV.partition_A(gdO);
//     Tensor tdVsdO = thr_mma_dV.partition_A(sdO);
//
//     Tensor tdVsdOt = thr_mma_dV.partition_B(sdOt);
//     Tensor tdVrdOt = thr_mma_dV.partition_fragment_B(sdOt);
//     Tensor tdVrdV_float = partition_fragment_C(tiled_mma_dV, Shape<Int<kBlockN>, Int<kHeadDim>>{});
//     Tensor tdVgdV = thr_mma_dV.partition_C(gdV);
//
//     // dK += dS^TQ
//     TiledMma_dK tiled_mma_dK;
//     ThrMMA thr_mma_dK = tiled_mma_dK.get_slice(threadIdx.x);
//     Tensor tdKsdSt = thr_mma_dK.partition_A(sdSt);
//     Tensor tdKsQt = thr_mma_dK.partition_B(sQt);
//     Tensor tdKrdK_float = partition_fragment_C(tiled_mma_dK, Shape<Int<kBlockN>, Int<kHeadDim>>{});
//     Tensor tdKgdK = thr_mma_dK.partition_C(gdK);


//     if (thread0()) {
//         print(gQ);
//         print("\n");
//         print(gK);
//         print("\n");
//         print(gV);
//         print("\n");
//         print(gL);
//         print("\n");
//         print(gD);
//         print("\n");
//         print(gdO);
//         print("\n");
//         print("gD[0] = %f\n", gD((0)));
//         print("\n");
//         print(sQ);
//         print("\n");
//         print(sK);
//         print("\n");
//         print(sV);
//         print("\n");
//         print(sdO);
//         print("\n");
//         print(tSgQ);
//         print("\n");
//         print(tSsQ);
//         print("\n");
//     }


    auto Q_TILE_MAX = size<3>(tSgQ);
    auto QK_BLOCK_MAX = size<2>(tSsK);

    // load K, V, dK, dV tiles

    copy(gmem_tiled_copy_QKV, tKgK, tKsK);
    //copy(tSgK, tSsK);
    //copy(gmem_tiled_copy_QKV, tVgV, tVsV);

    __syncthreads();

//     if (thread0()){
//         print_tensor(sK);
//         print("\n");
//     }

    //clear(tdVrdV_float);
    clear(tSrS_float);
    CUTE_NO_UNROLL
    for (int q_tile = 0; q_tile < Q_TILE_MAX; ++q_tile) {

        clear(tSrS_float);
       // clear(tdPrdP_float);

        // load gQ to sQ
//         copy(tSgQ(_,_,_,q_tile), tSsQ);
//         copy(tdVgdO(_,_,_,q_tile), tdVsdO);
        copy(gmem_tiled_copy_QKV, tQgQ(_,_,_,q_tile), tQsQ);
        //copy(gmem_tiled_copy_QKV, tdOgdO(_,_,_,q_tile), tdOsdO);

        // load gdQ to tdQrdQ
        //copy(tdQgdQ(_,_,_,q_tile), tdQrdQ);
//
//
//
        __syncthreads();

//         if (thread0()) {
//             print("tSsK\n");
//             print(tSsK);
//             print("\n");
//             print("====================");
//             print("tSsQ\n");
//             print(tSsQ);
//             print("\n");
//             print("====================");
//             print("\n");
//             print("gQ\n");
//             print(gQ);
//             print("====================");
//             print("\n");
//             print("tQgQ\n");
//             print(tQgQ);
//             print("\n");
//             print("====================");
//             print("\n");
//             print("sQ\n");
//             print(sQ);
//             print("\n");
//             print("####################");
//             print("\n");
//             //print(tdQgdQ_float);
//             //print_tensor(tdQrdQ);
//         }
//
//         if (thread0() && q_tile==1) {
//             print_tensor(gQ(_,_,1));
//             print("=========================\n");
//             print_tensor(sQ);
//             print("=========================\n");
//         }

//
//
        // compute S=QK^T
//         copy(tSsQ(_,_,0), tSrQ);
//         copy(tSsK(_,_,0), tSrK);
//         gemm(tiled_mma_S, tSrQ, tSrK, tSrS_float);


//         for (int qk_block = 0; qk_block < QK_BLOCK_MAX; qk_block++) {
//             gemm(tiled_mma_S, tSsQ(_,_,qk_block), tSsK(_,_,qk_block), tSrS_float);
//         }

        if (thread0()) {

            print("before\n");
            print_tensor(tSrS_float);
            print("\n=========================\n");
        }

        copy(tSsQ, tSrQ);
        copy(tSsK, tSrK);

        gemm(tiled_mma_S, tSrQ, tSrK, tSrS_float);

        if (thread0()) {
            print("after\n");
            print_tensor(tSrS_float);
            print("\n=========================\n");
        }
//
//         gemm(tiled_mma_dP, tdPsdO, tdPsV, tdPrdP_float);
//         //copy(tSrS_float, tSsS_float);
//
//         if (thread0()){
//             print_tensor(tSrS_float);
//             print("\n");
//         }

        __syncthreads();
//
//
//         // load rL, rD from gmem to rmem
// //         for (int i=0;i<2;i++) {
// //             for (int j=0;j<2;j++) {
// //                 rL[i][j] = gL((warp_offset + thread_offset + 8 * j + 32 * i), q_tile);
// //                 rD[i][j] = gD((warp_offset + thread_offset + 8 * j + 32 * i), q_tile);
// //             }
// //         }
//
//
// //         for (int i=0; i<2; i++) {
// //             rL[i] = gL((thread_row + 8 * i), q_tile);
// //             rD[i] = gD((thread_row + 8 * i), q_tile);
// //         }
//
//
//         // rescale S
//         for (int i=0;i< tSrS_float.size();i ++ ) {
//             tSrS_float[i] *= 1.0f / sqrtf(kHeadDim);
//         }
//
//         //copy(tSrS_float, tSsS_float);
//
//         Tensor tSrP_float = tSrS_float;
//         Tensor tdPrdS_float = tdPrdP_float;
//
//         // ptr[32b](0x7ea2408d8910) o ((_2,_2),_2,_2):((_1,_512),_2048,_32)
//         // for (int i=0;i<2;i++) {
//         //     for (int j=0;j<2;j++) {
//         //         print_tensor(tc(make_coord(_,j),i,_));
//         //     }
//
//         // compute P = exp(S-l)
//         for (int i=0; i<2; i++) {
//             for (int j=0;j<2;j++) {
//                 for (int k=0; k< tSrS_float(make_coord(_,j),i,_).size(); k++) {
//                     tSrP_float(make_coord(_,j),i,_)[k] = expf(tSrS_float(make_coord(_,j),i,_)[k] - rL[i][j]);
//                 }
//             }
//         }
//
//         // compute dS = P \circ (dP - D)
//         // tS has the same mma layout as tdP
//         for (int i=0; i<2; i++) {
//             for (int j=0;j<2;j++) {
//                 for (int k=0; k< tdPrdP_float(make_coord(_,j),i,_).size(); k++) {
//                     tdPrdS_float(make_coord(_,j),i,_)[k] = tSrP_float(make_coord(_,j),i,_)[k] * (tdPrdP_float(make_coord(_,j),i,_)[k] - rD[i][j]);
//                 }
//             }
//
//         }
//
// //         if (thread0()) {
// //             print_tensor(tSrS_float);
// //             print("\n");
// //             print_tensor(tSrP_float);
// //             print("\n");
// //             print_tensor(tdPrdP_float);
// //             print("\n");
// //             print_tensor(tdPrdS_float);
// //             print("\n");
// //         }
//
//
//
//         //convert P from fp32 to fp16
//         constexpr int num_element = decltype(size(tSrP_float))::value;
//
//         cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
//         auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tSrP_float.data()));
//
//         Tensor tSrP = make_tensor(make_rmem_ptr<half_t>(&frag), tSrP_float.layout());
//
//
//         // convert dS from fp32 to fp16
//         constexpr int num_element_dS = decltype(size(tdPrdS_float))::value;
//
//         cutlass::NumericArrayConverter<half_t, float, num_element_dS> convert_op_dS;
//         auto frag_dS = convert_op_dS(*reinterpret_cast<const cutlass::Array<float, num_element_dS> *>(tdPrdS_float.data()));
//
//         Tensor tdPrdS = make_tensor(make_rmem_ptr<half_t>(&frag_dS), tdPrdS_float.layout());
//
//
//         copy(tSrP, tSsP);
//         copy(tdPrdS, tdPsdS);
//
//
//         __syncthreads();
//
// //         if (thread0()) {
// //             for (int i=0;i<2;i++) {
// //                 for (int j=0;j<2;j++) {
// //                     print("%d\n", warp_offset + thread_offset + 8 * j + 32 * i);
// //                 }
// //             }
// //             print_tensor(sP);
// //             print("\n");
// //         }
//
//
//         // dV += P^TdO
//         gemm(tiled_mma_dV, tdVsPt, tdVsdOt, tdVrdV_float);
//
//         // dK += dS^TQ
//         gemm(tiled_mma_dK, tdKsdSt, tdKsQt, tdKrdK_float);
//
//
//         // dQ += dSK
//         //copy(tdQgdQ_float(_,_,_,0), tdQrdQ_float);
//
//         //print_tensor(tdQrdQ_float);
//
// //         gemm(tiled_mma_dQ, tdQsdS, tdQsKt, tdQrdQ_float);
// //
// //         if (thread0()) {
// //
// //             print(tdQrdQ_float);
// //         }
//         __syncthreads();
//
//         //convert dQ from float to fp16
//
//
    }
//
//
//     // dV
//     constexpr int num_element = decltype(size(tdVrdV_float))::value;
//
//     cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
//     auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tdVrdV_float.data()));
//
//     Tensor tdVrdV = make_tensor(make_rmem_ptr<half_t>(&frag), tdVrdV_float.layout());
//
//     copy(tdVrdV, tdVgdV);
//
//
//     // dK
//
//     // rescale by head dim
//     for (int i=0;i< tdKrdK_float.size();i ++ ) {
//         tdKrdK_float[i] *= 1.0f / sqrtf(kHeadDim);
//     }
//
//
//     constexpr int num_element_dK = decltype(size(tdKrdK_float))::value;
//
//     cutlass::NumericArrayConverter<half_t, float, num_element_dK> convert_op_dK;
//     auto frag_dK = convert_op_dK(*reinterpret_cast<const cutlass::Array<float, num_element_dK> *>(tdKrdK_float.data()));
//
//     Tensor tdKrdK = make_tensor(make_rmem_ptr<half_t>(&frag_dK), tdKrdK_float.layout());
//
//     copy(tdKrdK, tdKgdK);
//
//
// //     if (thread0()){
// //         for (int i =0;i<128;i++) {
// //             printf("i = %d, d[i] = %f\n", i, d_ptr[i]);
// //         }
// //     }


}


std::vector<torch::Tensor>
flash_bwd_v3(torch::Tensor q,
          torch::Tensor k,
          torch::Tensor v,
          torch::Tensor o,
          torch::Tensor l,
          torch::Tensor d_o,
          int batch_size, int seq_len, int num_heads, int head_dim)
{

    constexpr int kBlockM = 64;
    constexpr int kBlockN = 64;
    constexpr int kHeadDim = 128;

    torch::Tensor dq = torch::empty(q.sizes(), q.options().dtype(torch::kFloat16));
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

    // compute dK, dV
    dim3 dimGrid(batch_size, num_heads, seq_len / K_BLOCK_N);
    dim3 dimBlock(256);


    cudaFuncSetAttribute(compute_dq_dk_dv_kernel_v3, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    compute_dq_dk_dv_kernel_v3<<<dimGrid, dimBlock, maxbytes>>>(q_ptr,
                                            k_ptr,
                                            v_ptr,
                                            l_ptr,
                                            d_ptr,
                                            do_ptr,
                                            dk_ptr,
                                            dv_ptr,
                                            dq_ptr,
                                            batch_size, seq_len, num_heads, head_dim);

    return { dq, dk, dv };

}