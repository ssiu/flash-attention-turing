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

#include "kernel_traits.h"

using namespace cute;


__global__ __launch_bounds__(64)
void compute_dq_dk_dv_kernel_v1(
    half_t const* q_ptr,
    half_t const* k_ptr,
    half_t const* v_ptr,
    float const* l_ptr,
    half_t const* do_ptr,
//     half_t* d_ptr, // dO \circ O
    half_t* dq_ptr,
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

    extern __shared__ char smem_[];

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), SmemLayoutQ{});        // 8KB
    //Tensor sK = make_tensor(sQ.data() + kBlockM * kHeadDim, SmemLayoutKV{});
    Tensor sK = make_tensor(sQ.data() + size(sQ), SmemLayoutKV{});   // 8KB
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


    // dV += P^TdO
    TiledMma_dV tiled_mma_dV;
    ThrMMA thr_mma_dV = tiled_mma_dV.get_slice(threadIdx.x);
    Tensor tdVsPt = thr_mma_dV.partition_A(sPt);
    Tensor tdVgdO = thr_mma_dV.partition_A(gdO);
    Tensor tdVsdO = thr_mma_dV.partition_A(sdO);
    Tensor tdVsdOt = thr_mma_dV.partition_B(sdOt);
    Tensor tdVrdOt = thr_mma_dV.partition_fragment_B(sdOt);

    Tensor tdVrdV_float = partition_fragment_C(tiled_mma_dV, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    //Tensor tdVsdV = thr_mma_dV.partition_C(sdV);
    Tensor tdVgdV = thr_mma_dV.partition_C(gdV);

    // dP = dOV^T
    TiledMma_dP tiled_mma_dP;
    ThrMMA thr_mma_dP = tiled_mma_dP.get_slice(threadIdx.x);
    Tensor tdPgdO = thr_mma_dP.partition_A(gdO);
    Tensor tdPsdO = thr_mma_dP.partition_A(sdO);
    Tensor tdPgV = thr_mma_dP.partition_B(gV);
    Tensor tdPsV = thr_mma_dP.partition_B(sV);
    Tensor tdPrdP_float = partition_fragment_C(tiled_mma_dP, Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tdPsdS = thr_mma_dP.partition_C(sdS);


    auto Q_TILE_MAX = size<3>(tSgQ);

    // load K, V, dK, dV tiles
    copy(tSgK, tSsK);
    copy(tdPgV, tdPsV);

    //clear(tdVrdV_float);
    clear(tSrS_float);
    CUTE_NO_UNROLL
    for (int q_tile = 0; q_tile < Q_TILE_MAX; ++q_tile) {

        // load gQ to sQ
        copy(tSgQ(_,_,_,q_tile), tSsQ);
        copy(tdVgdO(_,_,_,q_tile), tdVsdO);


        __syncthreads();
        // compute S=QK^T
        gemm(tiled_mma_S, tSsQ, tSsK, tSrS_float);

        gemm(tiled_mma_dP, tdPsdO, tdPsV, tdPrdP_float);
        //copy(tSrS_float, tSsS_float);
        __syncthreads();


        // load rL, rD from gmem to rmem
        for (int i=0; i<2; i++) {
            rL[i] = gL((thread_row + 8 * i), q_tile);
        }


        // rescale S
        for (int i=0;i< tSrS_float.size();i ++ ) {
            tSrS_float[i] *= 1.0f / sqrtf(kHeadDim);
        }

        //copy(tSrS_float, tSsS_float);

        // compute P = exp(S-l)

        // P has size blockM x blockN, partitioned by mma_S
        // gL has size (32), need to figure the L_i for each S_ij

        for (int i=0; i<2; i++) {
            for (int j=0; j< tSrS_float(make_coord(_,i),_,_).size(); j++) {
                tSrS_float(make_coord(_,i),_,_)[j] = expf(tSrS_float(make_coord(_,i),_,_)[j] - rL[i]);
            }
        }

        // compute dS = P \circ (dP - D)
//         for (int i=0; i<2; i++) {
//             for (int j=0; j< tdPrdP_float(make_coord(_,i),_,_).size(); j++) {
//                 tdPrdP_float(make_coord(_,i),_,_)[j] = tSrS_float(make_coord(_,i),_,_)[j] * (tdPrdP_float(make_coord(_,i),_,_)[j] - rD[i]);
//             }
//         }




        //convert P from fp32 to fp16
        constexpr int num_element = decltype(size(tSrS_float))::value;

        cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
        auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tSrS_float.data()));

        Tensor tSrP = make_tensor(make_rmem_ptr<half_t>(&frag), tSrS_float.layout());
//
        copy(tSrP, tSsP);
//
        __syncthreads();

        clear(tSrS_float);

        gemm(tiled_mma_dV, tdVsPt, tdVsdOt, tdVrdV_float);

        __syncthreads();

    }

    constexpr int num_element = decltype(size(tdVrdV_float))::value;

    cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tdVrdV_float.data()));

    Tensor tdVrdV = make_tensor(make_rmem_ptr<half_t>(&frag), tdVrdV_float.layout());

//    copy(tdVrdV, tdVsdV);
    copy(tdVrdV, tdVgdV);


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

    half_t* q_ptr = reinterpret_cast<half_t*>(q.data_ptr());
    half_t* k_ptr = reinterpret_cast<half_t*>(k.data_ptr());
    half_t* v_ptr = reinterpret_cast<half_t*>(v.data_ptr());
    half_t* o_ptr = reinterpret_cast<half_t*>(o.data_ptr());
    float* l_ptr = reinterpret_cast<float*>(l.data_ptr());
    half_t* do_ptr = reinterpret_cast<half_t*>(d_o.data_ptr());

    half_t* dq_ptr = reinterpret_cast<half_t*>(dq.data_ptr());
    half_t* dk_ptr = reinterpret_cast<half_t*>(dk.data_ptr());
    half_t* dv_ptr = reinterpret_cast<half_t*>(dv.data_ptr());

    // compute dO \circ O
    //compute_dot_do_o


    dim3 dimGrid(batch_size, num_heads, seq_len / kBlockN);
    dim3 dimBlock(64);
    int maxbytes = 65536;


    // compute dQ, dK, dV

    cudaFuncSetAttribute(compute_dq_dk_dv_kernel_v1, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);


   compute_dq_dk_dv_kernel_v1<<<dimGrid, dimBlock, maxbytes>>>(q_ptr,
                                            k_ptr,
                                            v_ptr,
                                            l_ptr,
                                            do_ptr,
                                            dq_ptr,
                                            dk_ptr,
                                            dv_ptr,
                                            batch_size, seq_len, num_heads, head_dim);

    return { dq, dk, dv };

}