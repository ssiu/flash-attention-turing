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

#define FLOAT2(value) reinterpret_cast<float2*>(&(value))[0]

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


    // for (int i = 0; i < 4; i++) {
    //     FLOAT4(tQ[i][0]) = FLOAT4(gQ(thread_row + i, thread_col));
    //     FLOAT4(tQ[i+4][0]) = FLOAT4(gQ(thread_row + i + 8, thread_col));
    // }



    // load O to register
//     FLOAT2(rO[0]) = FLOAT2(o_ptr[do_o_offset + thread_row + thread_col]);
//     FLOAT2(rdO[0]) = FLOAT2(do_ptr[do_o_offset + thread_row + thread_col]);

    for (int i=0;i<4;i++) {
        rO[i] = o_ptr[do_o_offset + thread_row * num_heads * head_dim + thread_col + i];
        rdO[i] = do_ptr[do_o_offset + thread_row * num_heads * head_dim + thread_col + i];

//         if (warp_id == 1 && lane_id == 0 && blockIdx.z == 0) {
//             printf("do_o_offset = %d, thread_row = %d, thread_col = %d, addr = %d\n", do_o_offset, thread_row, thread_col, do_o_offset + thread_row * num_heads * head_dim + thread_col + i);
//             printf("thread reduction, sum is %f\n",  sum);
//             for (int i=0;i<4;i++) {
//                 printf("i = %d, rO[i] = %f, rdO[i] = %f\n", i, static_cast<float>(rO[i]), static_cast<float>(rdO[i]));
//             }
//            //d_ptr[0] = sum;
//         }

    }

//     if (thread0()) {
//         for (int i=0;i<128;i++) {
//             printf("row = 0, col = %d, o = %f, do = %f\n", i, static_cast<float>(o_ptr[i]), static_cast<float>(do_ptr[i]));
//         }
//     }


//     if (blockIdx.x == 0 and thread_id < 32) {
//         for (int i=0;i<4;i++) {
//             printf("thread id = %d, i = %d, rdO = %f, rO = %f\n", thread_id, i, static_cast<float>(rO[i]), static_cast<float>(rdO[i]));
//         }
//     }




//     if (thread0()) {
//         for (int i=0;i<4;i++) {
//             printf("%f\n", static_cast<float>(rO[i]));
//         }
//     }





    // thread reduction
    for (int i=0;i<4;i ++) {
//         sum += __half2float(rO[i]) * __half2float(rdO[i]);
        sum += static_cast<float>(rO[i]) * static_cast<float>(rdO[i]);

    }







    // warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
       d_ptr[d_offset + thread_row] = sum;
       //printf("warp id = %d, sum is %f\n", warp_id, sum);
       //d_ptr[0] = sum;
    }

//     if (thread0()) {
//         for (int i=0; i<4; i++) {
//             for (int j=0;j<128;j++) {
//                 print("i = %d, j = %d, do = %f\n", i, j, static_cast<float>(do_ptr[128 * i + j]));
//             }
//         }
//     }


}



__global__ __launch_bounds__(64)
void compute_dq_dk_dv_kernel_v1(
    half_t const* q_ptr,
    half_t const* k_ptr,
    half_t const* v_ptr,
    float const* l_ptr,
    float const* d_ptr,
    half_t const* do_ptr,
    half_t* dk_ptr,
    half_t* dv_ptr,
    int batch_size, int seq_len, int num_heads, int head_dim
)
{   
    
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
    
    using TiledMma_S = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_2,_1,_1>>,
        Tile<_32, _32, _8>>;

    using TiledMma_dP = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_2,_1,_1>>,
        Tile<_32, _32, _8>>;

    using TiledMma_dV = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_2,_1,_1>>,
        Tile<_32, _128, _8>>;

    using TiledMma_dK = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_2,_1,_1>>,
        Tile<_32, _128, _8>>;


    using TiledMma_dQ = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_2,_1,_1>>,
        Tile<_32, _128, _8>>;


    using Gmem_copy_struct = AutoVectorizingCopyWithAssumedAlignment<128>;

    using GmemLayoutAtomQKV = Layout<Shape <_8, _8>, Stride<_8, _1>>;

    using GmemTiledCopyQKV = decltype(
                make_tiled_copy(Copy_Atom<Gmem_copy_struct, half_t>{},
                                GmemLayoutAtomQKV{},
                                Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read


    using SmemLayoutAtom = decltype(
                    Layout<Shape<_32, _32>,
                    Stride<_32, _1>>{});

    using SmemLayoutAtomTranposed = decltype(
                    Layout<Shape<_32, _32>,
                    Stride<_1, _32>>{});
    
    using SmemLayoutQ = decltype(
                            Layout<Shape<_32, _128>,
                            Stride<_128, _1>>{});

    using SmemLayoutQTransposed = decltype(
                                      Layout<Shape<_128, _32>,
                                      Stride<_1, _128>>{});



    using SmemLayoutKV = decltype(
           Layout<Shape<_32, _128>,
           Stride<_128, _1>>{});

    using SmemLayoutKVTransposed = decltype(
           Layout<Shape<_128, _32>,
           Stride<_1, _128>>{});

    constexpr int kBlockM = 32;
    constexpr int kBlockN = 32;
    constexpr int kHeadDim = 128;
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

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), SmemLayoutQ{});        // 8KB
    Tensor sQt = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), SmemLayoutQTransposed{});
    //Tensor sK = make_tensor(sQ.data() + kBlockM * kHeadDim, SmemLayoutKV{});
    Tensor sK = make_tensor(sQ.data() + size(sQ), SmemLayoutKV{});   // 8KB
    Tensor sKt = make_tensor(sQ.data() + size(sQ), SmemLayoutKVTransposed{});   // 8KB

    Tensor sV = make_tensor(sK.data() + size(sK), SmemLayoutKV{});

    Tensor sdO = make_tensor(sV.data() + size(sV), SmemLayoutQ{});                            // 8KB
    Tensor sdOt = make_tensor(sV.data() + size(sV), SmemLayoutQTransposed{});                 // 8KB

    Tensor sP = make_tensor(sdO.data() + size(sdO), SmemLayoutAtom{});                         // 2KB
    Tensor sPt = make_tensor(sdO.data() + size(sdO), SmemLayoutAtomTranposed{});               // 2KB

    Tensor sdS = make_tensor(sP.data() + size(sP), SmemLayoutAtom{});     // 2KB
    Tensor sdSt = make_tensor(sP.data() + size(sP), SmemLayoutAtomTranposed{});     // 2KB

    //Tensor sdV = make_tensor(sdS.data() + size(sdS), SmemLayoutKV{});                            // 2KB

    //int total_bytes_for_half = cosize_v<SmemLayoutQ> * 2 + cosize_v<SmemLayoutQTransposed> + cosize_v<SmemLayoutKV> * 2 + cosize_v<SmemLayoutAtom> + cosize_v<SmemLayoutAtomTranposed>;

    // only
    //Tensor sS = make_tensor(make_smem_ptr(reinterpret_cast<float*>(&smem_[0])), SmemLayoutAtom{});      // 2KB

//     if (thread0()){
//         printf("sdV size %d\n", size(sdV));
//     }

    int thread_id = threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    int thread_row = warp_id * 16 + lane_id / 4;

    float rL[2];
    float rD[2];

    // Copy operation
    GmemTiledCopyQKV gmem_tiled_copy_QKV;

    ThrCopy thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(threadIdx.x);

    Tensor tQgQ = thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = thr_copy_QKV.partition_D(sQ);

    Tensor tKgK = thr_copy_QKV.partition_S(gK);
    Tensor tKsK = thr_copy_QKV.partition_D(sK);

    Tensor tVgV = thr_copy_QKV.partition_S(gV);
    Tensor tVsV = thr_copy_QKV.partition_D(sV);

    Tensor tdOgdO = thr_copy_QKV.partition_S(gdO);
    Tensor tdOsdO = thr_copy_QKV.partition_D(sdO);


    // S = QK^T
    TiledMma_S tiled_mma_S;
    ThrMMA thr_mma_S = tiled_mma_S.get_slice(threadIdx.x);
    Tensor tSgQ = thr_mma_S.partition_A(gQ);
    Tensor tSsQ = thr_mma_S.partition_A(sQ);
    Tensor tSgK = thr_mma_S.partition_B(gK);
    Tensor tSsK = thr_mma_S.partition_B(sK);
    Tensor tSrS_float = partition_fragment_C(tiled_mma_S, Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tSsP = thr_mma_S.partition_C(sP);
    //Tensor tSsS_float = thr_mma_S.partition_C(sS);


    // dP = dOV^T
    TiledMma_dP tiled_mma_dP;
    ThrMMA thr_mma_dP = tiled_mma_dP.get_slice(threadIdx.x);
    Tensor tdPgdO = thr_mma_dP.partition_A(gdO);
    Tensor tdPsdO = thr_mma_dP.partition_A(sdO);
    Tensor tdPgV = thr_mma_dP.partition_B(gV);
    Tensor tdPsV = thr_mma_dP.partition_B(sV);
    Tensor tdPrdP_float = partition_fragment_C(tiled_mma_dP, Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tdPsdS = thr_mma_dP.partition_C(sdS);


    // dV += P^TdO
    TiledMma_dV tiled_mma_dV;
    ThrMMA thr_mma_dV = tiled_mma_dV.get_slice(threadIdx.x);
    Tensor tdVsPt = thr_mma_dV.partition_A(sPt);
    // for copying dO from gmem to smem
    Tensor tdVgdO = thr_mma_dV.partition_A(gdO);
    Tensor tdVsdO = thr_mma_dV.partition_A(sdO);

    Tensor tdVsdOt = thr_mma_dV.partition_B(sdOt);
    Tensor tdVrdOt = thr_mma_dV.partition_fragment_B(sdOt);
    Tensor tdVrdV_float = partition_fragment_C(tiled_mma_dV, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor tdVgdV = thr_mma_dV.partition_C(gdV);

    // dK += dS^TQ
    TiledMma_dK tiled_mma_dK;
    ThrMMA thr_mma_dK = tiled_mma_dK.get_slice(threadIdx.x);
    Tensor tdKsdSt = thr_mma_dK.partition_A(sdSt);
    Tensor tdKsQt = thr_mma_dK.partition_B(sQt);
    Tensor tdKrdK_float = partition_fragment_C(tiled_mma_dK, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor tdKgdK = thr_mma_dK.partition_C(gdK);



    auto Q_TILE_MAX = size<3>(tSgQ);

    // load K, V, dK, dV tiles
//     copy(tSgK, tSsK);
//     copy(tdPgV, tdPsV);

    copy(gmem_tiled_copy_QKV, tKgK, tKsK);
    copy(gmem_tiled_copy_QKV, tVgV, tVsV);


    //clear(tdVrdV_float);
    clear(tSrS_float);
    CUTE_NO_UNROLL
//     if (thread0()) {
//         printf("gdQ is %f\n", dq_ptr[0]);
//         print("\n");
//         print(gdQ);
//         print("\n");
//         print(tdQgdQ_float);
//         print("\n");
//         print(tdQgdQ_float(_,_,_,0));
//         print("\n");
//         print(tdQrdQ_float);
//         print("\n");
//         print_tensor(tdQrdQ_float);
//         copy(tdQgdQ_float(_,_,_,0), tdQrdQ_float);
//         print_tensor(tdQrdQ_float);
//     }



    for (int q_tile = 0; q_tile < Q_TILE_MAX; ++q_tile) {

        clear(tSrS_float);
        clear(tdPrdP_float);

        // load gQ to sQ
//         copy(tSgQ(_,_,_,q_tile), tSsQ);
//         copy(tdVgdO(_,_,_,q_tile), tdVsdO);
        copy(gmem_tiled_copy_QKV, tQgQ(_,_,_,q_tile), tQsQ);
        copy(gmem_tiled_copy_QKV, tdOgdO(_,_,_,q_tile), tdOsdO);

        // load gdQ to tdQrdQ
        //copy(tdQgdQ(_,_,_,q_tile), tdQrdQ);



//         if (thread0()) {
//             print(tdQrdQ_float);
//             //print(tdQgdQ_float);
//             //print_tensor(tdQrdQ);
//         }

        __syncthreads();
        // compute S=QK^T
        gemm(tiled_mma_S, tSsQ, tSsK, tSrS_float);

        gemm(tiled_mma_dP, tdPsdO, tdPsV, tdPrdP_float);
        //copy(tSrS_float, tSsS_float);
        __syncthreads();


        // load rL, rD from gmem to rmem
        for (int i=0; i<2; i++) {
            rL[i] = gL((thread_row + 8 * i), q_tile);
            rD[i] = gD((thread_row + 8 * i), q_tile);
        }


        // rescale S
        for (int i=0;i< tSrS_float.size();i ++ ) {
            tSrS_float[i] *= 1.0f / sqrtf(kHeadDim);
        }

        //copy(tSrS_float, tSsS_float);

        // compute P = exp(S-l)

        // P has size blockM x blockN, partitioned by mma_S
        // gL has size (32), need to figure the L_i for each S_ij

        Tensor tSrP_float = tSrS_float;
        Tensor tdPrdS_float = tdPrdP_float;

        for (int i=0; i<2; i++) {
            for (int j=0; j< tSrS_float(make_coord(_,i),_,_).size(); j++) {
                tSrP_float(make_coord(_,i),_,_)[j] = expf(tSrS_float(make_coord(_,i),_,_)[j] - rL[i]);
            }
        }

        // compute dS = P \circ (dP - D)
        // tS has the same mma layout as tdP
        for (int i=0; i<2; i++) {
            for (int j=0; j< tdPrdP_float(make_coord(_,i),_,_).size(); j++) {
                tdPrdS_float(make_coord(_,i),_,_)[j] = tSrP_float(make_coord(_,i),_,_)[j] * (tdPrdP_float(make_coord(_,i),_,_)[j] - rD[i]);
            }
        }

//         if (thread0()) {
//             print_tensor(tSrS_float);
//             print("\n");
//             print_tensor(tSrP_float);
//             print("\n");
//             print_tensor(tdPrdP_float);
//             print("\n");
//             print_tensor(tdPrdS_float);
//             print("\n");
//         }



        //convert P from fp32 to fp16
        constexpr int num_element = decltype(size(tSrP_float))::value;

        cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
        auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tSrP_float.data()));

        Tensor tSrP = make_tensor(make_rmem_ptr<half_t>(&frag), tSrP_float.layout());


        // convert dS from fp32 to fp16
        constexpr int num_element_dS = decltype(size(tdPrdS_float))::value;

        cutlass::NumericArrayConverter<half_t, float, num_element_dS> convert_op_dS;
        auto frag_dS = convert_op_dS(*reinterpret_cast<const cutlass::Array<float, num_element_dS> *>(tdPrdS_float.data()));

        Tensor tdPrdS = make_tensor(make_rmem_ptr<half_t>(&frag_dS), tdPrdS_float.layout());


        copy(tSrP, tSsP);
        copy(tdPrdS, tdPsdS);


        __syncthreads();



        // dV += P^TdO
        gemm(tiled_mma_dV, tdVsPt, tdVsdOt, tdVrdV_float);

        // dK += dS^TQ
        gemm(tiled_mma_dK, tdKsdSt, tdKsQt, tdKrdK_float);


        // dQ += dSK
        //copy(tdQgdQ_float(_,_,_,0), tdQrdQ_float);

        //print_tensor(tdQrdQ_float);

//         gemm(tiled_mma_dQ, tdQsdS, tdQsKt, tdQrdQ_float);
//
//         if (thread0()) {
//
//             print(tdQrdQ_float);
//         }
        __syncthreads();

        //convert dQ from float to fp16


    }


    // dV
    constexpr int num_element = decltype(size(tdVrdV_float))::value;

    cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tdVrdV_float.data()));

    Tensor tdVrdV = make_tensor(make_rmem_ptr<half_t>(&frag), tdVrdV_float.layout());

    copy(tdVrdV, tdVgdV);


    // dK

    // rescale by head dim
    for (int i=0;i< tdKrdK_float.size();i ++ ) {
        tdKrdK_float[i] *= 1.0f / sqrtf(kHeadDim);
    }


    constexpr int num_element_dK = decltype(size(tdKrdK_float))::value;

    cutlass::NumericArrayConverter<half_t, float, num_element_dK> convert_op_dK;
    auto frag_dK = convert_op_dK(*reinterpret_cast<const cutlass::Array<float, num_element_dK> *>(tdKrdK_float.data()));

    Tensor tdKrdK = make_tensor(make_rmem_ptr<half_t>(&frag_dK), tdKrdK_float.layout());

    copy(tdKrdK, tdKgdK);


//     if (thread0()){
//         for (int i =0;i<128;i++) {
//             printf("i = %d, d[i] = %f\n", i, d_ptr[i]);
//         }
//     }


}


__global__ __launch_bounds__(64)
void compute_dq_kernel_v1(
    half_t const* q_ptr,
    half_t const* k_ptr,
    half_t const* v_ptr,
    float const* l_ptr,
    float const* d_ptr,
    half_t const* do_ptr,
    half_t* dq_ptr,
    int batch_size, int seq_len, int num_heads, int head_dim
)
{
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;

    using TiledMma_dP = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_2,_1,_1>>,
        Tile<_32, _32, _8>>;

    using Gmem_copy_struct = AutoVectorizingCopyWithAssumedAlignment<128>;

    using GmemLayoutAtomQKV = Layout<Shape <_8, _8>, Stride<_8, _1>>;

    using GmemTiledCopyQKV = decltype(
                make_tiled_copy(Copy_Atom<Gmem_copy_struct, half_t>{},
                                GmemLayoutAtomQKV{},
                                Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read


    using SmemLayoutAtom = decltype(
                    Layout<Shape<_32, _32>,
                    Stride<_32, _1>>{});

    using SmemLayoutAtomTranposed = decltype(
                    Layout<Shape<_32, _32>,
                    Stride<_1, _32>>{});

    using SmemLayoutQ = decltype(
                            Layout<Shape<_32, _128>,
                            Stride<_128, _1>>{});

    using SmemLayoutQTransposed = decltype(
                                      Layout<Shape<_128, _32>,
                                      Stride<_1, _128>>{});



    using SmemLayoutKV = decltype(
           Layout<Shape<_32, _128>,
           Stride<_128, _1>>{});

    using SmemLayoutKVTransposed = decltype(
           Layout<Shape<_128, _32>,
           Stride<_1, _128>>{});

    constexpr int kBlockM = 32;
    constexpr int kBlockN = 32;
    constexpr int kHeadDim = 128;



//     gmem_ptr[16b](0x2a78cd208000) o (_32,_128,4):(128,_1,4096)
//     gmem_ptr[16b](0x2a78cd210000) o (_32,_128,4):(128,_1,4096)
//     gmem_ptr[32b](0x2a78cd240200) o ((_32)):((_1))
//     gmem_ptr[16b](0x2a78cd220200) o (_32,_128):(128,_1)
//     gmem_ptr[16b](0x2a78cd228200) o (_32,_128):(128,_1)

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

    if (thread0()) {
        print(gK);
        print("\n");
        print(gV);
        print("\n");
        print(gD);
        print("\n");
        print(gdO);
        print("\n");
        print(gdQ);
        print("\n");
        print("gD[0] = %f\n", gD((0)));
        print("\n");
    }

    extern __shared__ char smem_[];

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), SmemLayoutQ{});        // 8KB
    Tensor sQt = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), SmemLayoutQTransposed{});
    Tensor sK = make_tensor(sQ.data() + size(sQ), SmemLayoutKV{});   // 8KB
    Tensor sKt = make_tensor(sQ.data() + size(sQ), SmemLayoutKVTransposed{});   // 8KB

    Tensor sV = make_tensor(sK.data() + size(sK), SmemLayoutKV{});

    Tensor sdO = make_tensor(sV.data() + size(sV), SmemLayoutQ{});                            // 8KB
    Tensor sdOt = make_tensor(sV.data() + size(sV), SmemLayoutQTransposed{});                 // 8KB

    Tensor sP = make_tensor(sdO.data() + size(sdO), SmemLayoutAtom{});                         // 2KB
    Tensor sPt = make_tensor(sdO.data() + size(sdO), SmemLayoutAtomTranposed{});               // 2KB

    Tensor sdS = make_tensor(sP.data() + size(sP), SmemLayoutAtom{});     // 2KB
    Tensor sdSt = make_tensor(sP.data() + size(sP), SmemLayoutAtomTranposed{});     // 2KB

    int thread_id = threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    int thread_row = warp_id * 16 + lane_id / 4;


    float rD[2];

    // Copy operation
    GmemTiledCopyQKV gmem_tiled_copy_QKV;

    ThrCopy thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(threadIdx.x);

    Tensor tKgK = thr_copy_QKV.partition_S(gK);
    Tensor tKsK = thr_copy_QKV.partition_D(sK);

    Tensor tVgV = thr_copy_QKV.partition_S(gV);
    Tensor tVsV = thr_copy_QKV.partition_D(sV);

    Tensor tdOgdO = thr_copy_QKV.partition_S(gdO);
    Tensor tdOsdO = thr_copy_QKV.partition_D(sdO);

    // S = QK^T
    TiledMma_S tiled_mma_S;
    ThrMMA thr_mma_S = tiled_mma_S.get_slice(threadIdx.x);
    Tensor tSsQ = thr_mma_S.partition_A(sQ);
    Tensor tSsK = thr_mma_S.partition_B(sK);
    Tensor tSrS_float = partition_fragment_C(tiled_mma_S, Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tSsP = thr_mma_S.partition_C(sP);


    // dP = dOV^T
    TiledMma_dP tiled_mma_dP;
    ThrMMA thr_mma_dP = tiled_mma_dP.get_slice(threadIdx.x);
    Tensor tdPgdO = thr_mma_dP.partition_A(gdO);
    Tensor tdPsdO = thr_mma_dP.partition_A(sdO);
    Tensor tdPgV = thr_mma_dP.partition_B(gV);
    Tensor tdPsV = thr_mma_dP.partition_B(sV);
    Tensor tdPrdP_float = partition_fragment_C(tiled_mma_dP, Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tdPsdS = thr_mma_dP.partition_C(sdS);


    // dQ = dSK
    TiledMma_dQ tiled_mma_dQ;
    ThrMMA thr_mma_dQ = tiled_mma_dQ.get_slice(threadIdx.x);
    Tensor tdQsdS = thr_mma_dQ.partition_A(sdS);
    Tensor tdQsKt = thr_mma_dQ.partition_B(sKt);
    Tensor tdQrdQ_float = partition_fragment_C(tiled_mma_dQ, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor tdQgdQ = thr_mma_dP.partition_C(gdQ);



    auto KV_TILE_MAX = size<3>(tKgK);

    copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
    copy(gmem_tiled_copy_QKV, tdOgdO, tdOsdO);


    CUTE_NO_UNROLL
    for (int kv_tile = 0; kv_tile < KV_TILE_MAX; ++kv_tile) {
        clear(tSrS_float);
        clear(tdPrdP_float);

        copy(gmem_tiled_copy_QK, tKgK(_,_,_,kv_tile), tKsK);
        copy(gmem_tiled_copy_QK, tVgV(_,_,_,kv_tile), tVsV);

        __syncthreads();

        gemm(tiled_mma_S, tSsQ, tSsK, tSrS_float);
        gemm(thr_mma_dP, tdPsdO, tdPsV, tdPrdP_float);


        __syncthreads();

        // load rL, rD from gmem to rmem
        for (int i=0; i<2; i++) {
            rL[i] = gL((thread_row + 8 * i), q_tile);
            rD[i] = gD((thread_row + 8 * i), q_tile);
        }


        // rescale S
        for (int i=0;i< tSrS_float.size();i ++ ) {
            tSrS_float[i] *= 1.0f / sqrtf(kHeadDim);
        }

        //copy(tSrS_float, tSsS_float);

        // compute P = exp(S-l)

        // P has size blockM x blockN, partitioned by mma_S
        // gL has size (32), need to figure the L_i for each S_ij

        Tensor tSrP_float = tSrS_float;
        Tensor tdPrdS_float = tdPrdP_float;

        for (int i=0; i<2; i++) {
            for (int j=0; j< tSrS_float(make_coord(_,i),_,_).size(); j++) {
                tSrP_float(make_coord(_,i),_,_)[j] = expf(tSrS_float(make_coord(_,i),_,_)[j] - rL[i]);
            }
        }

        // compute dS = P \circ (dP - D)
        // tS has the same mma layout as tdP
        for (int i=0; i<2; i++) {
            for (int j=0; j< tdPrdP_float(make_coord(_,i),_,_).size(); j++) {
                tdPrdS_float(make_coord(_,i),_,_)[j] = tSrP_float(make_coord(_,i),_,_)[j] * (tdPrdP_float(make_coord(_,i),_,_)[j] - rD[i]);
            }
        }


        //convert P from fp32 to fp16
        constexpr int num_element = decltype(size(tSrP_float))::value;

        cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
        auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tSrP_float.data()));

        Tensor tSrP = make_tensor(make_rmem_ptr<half_t>(&frag), tSrP_float.layout());


        // convert dS from fp32 to fp16
        constexpr int num_element_dS = decltype(size(tdPrdS_float))::value;

        cutlass::NumericArrayConverter<half_t, float, num_element_dS> convert_op_dS;
        auto frag_dS = convert_op_dS(*reinterpret_cast<const cutlass::Array<float, num_element_dS> *>(tdPrdS_float.data()));

        Tensor tdPrdS = make_tensor(make_rmem_ptr<half_t>(&frag_dS), tdPrdS_float.layout());


        copy(tSrP, tSsP);
        copy(tdPrdS, tdPsdS);


        __syncthreads();


        gemm(tiled_mma_dQ, tdQsdS, tdQsKt, tdQrdQ);

        __syncthreads();


    }



    // rescale by head dim
    for (int i=0;i< tdQrdQ_float.size();i ++ ) {
        tdQrdQ_float[i] *= 1.0f / sqrtf(kHeadDim);
    }


    constexpr int num_element_dQ = decltype(size(tdQrdQ_float))::value;

    cutlass::NumericArrayConverter<half_t, float, num_element_dQ> convert_op_dQ;
    auto frag_dQ = convert_op_dQ(*reinterpret_cast<const cutlass::Array<float, num_element_dQ> *>(tdQrdQ_float.data()));

    Tensor tdQrdQ = make_tensor(make_rmem_ptr<half_t>(&frag_dQ), tdQrdQ_float.layout());

    copy(tdQrdQ, tdQgdQ);

}


std::vector<torch::Tensor>
flash_bwd_v1(torch::Tensor q,
          torch::Tensor k,
          torch::Tensor v,
          torch::Tensor o,
          torch::Tensor l,
          torch::Tensor d_o,
          int batch_size, int seq_len, int num_heads, int head_dim)
{

    constexpr int kBlockM = 32;
    constexpr int kBlockN = 32;
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

    // compute dQ
    dim3 dimGrid_dq(batch_size, num_heads, seq_len / kBlockM);
    dim3 dimBlock_dq(64);
    int maxbytes = 65536;


    cudaFuncSetAttribute(compute_dq_kernel_v1, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    compute_dq_kernel_v1<<<dimGrid_dq, dimBlock_dq, maxbytes>>>(
                                                            q_ptr,
                                                            k_ptr,
                                                            v_ptr,
                                                            l_ptr,
                                                            d_ptr,
                                                            do_ptr,
                                                            dq_ptr,
                                                            batch_size, seq_len, num_heads, head_dim);


    // compute dK, dV
    dim3 dimGrid(batch_size, num_heads, seq_len / kBlockN);
    dim3 dimBlock(64);


    cudaFuncSetAttribute(compute_dq_dk_dv_kernel_v1, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    compute_dq_dk_dv_kernel_v1<<<dimGrid, dimBlock, maxbytes>>>(q_ptr,
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