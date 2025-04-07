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

// #define FLOAT4(value) reinterpret_cast<float4*>(&(value))[0]

// template <typename Kernel_traits>
// __global__ __launch_bounds__(1024)
// void compute_dot_do_o_kernel(
//     half_t const* o,
//     half_t const* d_o,
//     int batch_size, int seq_len, int num_heads, int head_dim
// )
// {
//     // dout (B, T, NH, HS)
//     // out (B, T, NH, HS)
//     // d (B, T, NH)
//
//     // blockDim.x = NH
//     // blockDim.y = T / 256
//     // blockDim.z = B
//
//     // Each half-warps compute 4 rows,
//     // so each warp computes 8 rows
//     // We use 1024 threads = 32 warps per block, so each block computes 256 rows
//     // so we have B * T / 256 * NH blocks
//
//     int o_global_offset = blockIdx.z * T * NH * HS + blockIdx.y * 256 * NH * HS + blockIdx.x * HS;
//     int d_global_offset = blockIdx.z * T * NH + blockIdx.y * 256 * NH + blockIdx.x;
//
//     int warp_id = threadIdx.x / 32;
//     int lane_id = threadIdx.x % 32;
//
//     unsigned mask = (lane_id < 16) ? 0xFFFF : 0xFFFF0000; // Mask for the two halves
//
//     float* gO = &out[o_global_offset];
//     float* gdO = &dout[o_global_offset];
//     float* gD = &d[d_global_offset];
//
//     int thread_row = warp_id * 8 + (lane_id / 16) * 4;
//     int thread_col = (lane_id % 16) * 4;
//
//     float tO[4][4];
//     float tdO[4][4];
//     float sum[4] = {0.0f};
//
//     for (int i=0;i<4;i++){
//         FLOAT4(tO[i][0]) = FLOAT4(gO(thread_row + i, thread_col));
//         FLOAT4(tdO[i][0]) = FLOAT4(gdO(thread_row + i, thread_col));
//     }
//
//     // inter-thread reduction
//     for (int i = 0; i < 4; i++) {
//         for (int j = 0; j < 4;j++) {
//             sum[i] += tO[i][j] * tdO[i][j];
//         }
//     }
//
//     // inter-warp reduction
//     for (int i=0; i < 4; i++) {
//         for (int offset = 8; offset > 0; offset /= 2) {
//            sum[i] += __shfl_down_sync(mask, sum[i], offset);
//         }
//     }
//
//     if (lane_id == 0 || lane_id == 16) {
//         for (int i=0; i<4; i++) {
//             gD(thread_row + i) = sum[i];
//         }
//     }
// }



template <typename Kernel_traits>
__global__ __launch_bounds__(64)
void compute_dq_dk_dv_kernel(
    half_t const* q_ptr,
    half_t const* k_ptr,
    half_t const* v_ptr,
    float* l_ptr,
    half_t* do_ptr,
//     half_t* d_ptr, // dO \circ O
    half_t const* dq_ptr,
    half_t const* dk_ptr,
    half_t const* dv_ptr,
    int batch_size, int seq_len, int num_heads, int head_dim
)
{

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
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

    // dO
    Tensor mdO = make_tensor(make_gmem_ptr(do_ptr),
                             make_shape(batch_size, seq_len, num_heads, head_dim),
                             make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gdO = local_tile(mdO(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(_, 0));



    // L = m + log l
    Tensor mL = make_tensor(make_gmem_ptr(l_ptr),
                             make_shape(batch_size, num_heads, seq_len),
                             make_stride(seq_len * num_heads,  seq_len, Int<1>{}));

    Tensor gL = local_tile(mL(blockIdx.x, blockIdx.y, _), Shape<Int<kBlockM>>{},
                           make_coord(_));

    // D = dO \circ O
//     Tensor mD = make_tensor(make_gmem_ptr(d_ptr),
//                              make_shape(batch_size, seq_len, num_heads),
//                              make_stride(seq_len * num_heads, Int<1>{}, seq_len ));
//
//     Tensor gD = local_tile(mD(blockIdx.x, _, blockIdx.y), Shape<Int<kBlockM>>{},
//                            make_coord(_));

    // dQ
    Tensor mdQ = make_tensor(make_gmem_ptr(dq_ptr),
                                make_shape(batch_size, seq_len, num_heads, head_dim),
                                make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gdQ = local_tile(mQ(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(_, 0));

    // dK
    Tensor mdK = make_tensor(make_gmem_ptr(dk_ptr),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gdK = local_tile(mK(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(blockIdx.z, 0));

    // dV
    Tensor mdV = make_tensor(make_gmem_ptr(dv_ptr),
                            make_shape(batch_size, seq_len, num_heads, head_dim),
                            make_stride(seq_len * num_heads * head_dim, num_heads * head_dim, head_dim, Int<1>{}));

    Tensor gdV = local_tile(mV(blockIdx.x, _, blockIdx.y, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(blockIdx.z, 0));



    extern __shared__ char smem_[];


    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(&smem_[0])), typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + kBlockM * kHeadDim, typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + kBlockN * kHeadDim, typename Kernel_traits::SmemLayoutKV{});
    Tensor sdO = make_tensor(sV.data() + kBlockN * kHeadDim, typename Kernel_traits::SmemLayoutQ{});
    Tensor sdOt = make_tensor(sV.data() + kBlockN * kHeadDim, typename Kernel_traits::SmemLayoutQTransposed{});

    Tensor sdQ = make_tensor(sdO.data() + kBlockM * kHeadDim, typename Kernel_traits::SmemLayoutQ{});
    Tensor sdK = make_tensor(sdQ.data() + kBlockM * kHeadDim, typename Kernel_traits::SmemLayoutKV{});
    Tensor sdV = make_tensor(sdK.data() + kBlockN * kHeadDim, typename Kernel_traits::SmemLayoutKV{});

    Tensor sP = make_tensor(sdV.data() + kBlockN * kHeadDim, typename Kernel_traits::SmemLayoutAtom{});
    Tensor sPt = make_tensor(sdV.data() + kBlockN * kHeadDim, typename Kernel_traits::SmemLayoutAtomTranposed{});




    int thread_id = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int thread_row = warp_id * 16 + thread_id / 4;


    float rL[2];
//     float rD[2];

    // gmem load
    typename Kernel_traits::GmemTiledCopy gmem_tiled_copy;

    ThrCopy thr_copy = gmem_tiled_copy.get_slice(threadIdx.x);

    Tensor tQgQ = thr_copy.partition_S(gQ);
    Tensor tQsQ = thr_copy.partition_D(sQ);

    Tensor tKgK = thr_copy.partition_S(gK);
    Tensor tKsK = thr_copy.partition_D(sK);

    Tensor tVgV = thr_copy.partition_S(gV);
    Tensor tVsV = thr_copy.partition_D(sV);

    Tensor tdOgdO = thr_copy.partition_S(gdO);
    Tensor tdOsdO = thr_copy.partition_D(sdO);



    // gmem store
    Tensor tQgdQ = thr_copy.partition_S(gdQ);
    Tensor tQsdQ = thr_copy.partition_D(sdQ);

    Tensor tKgdK = thr_copy.partition_S(gdK);
    Tensor tKsdK = thr_copy.partition_D(sdK);

    Tensor tVgdV = thr_copy.partition_S(gdV);
    Tensor tVsdV = thr_copy.partition_D(sdV);


    typename Kernel_traits::TiledMma tiled_mma;

    // S = QK^T
    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tSsQ = thr_mma.partition_A(sQ);
    Tensor tSsK = thr_mma.partition_B(sK);
    Tensor tSrS_float = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});

    Tensor tSsP = thr_mma.partition_A(sP);


    typename Kernel_traits::TiledMma_dV tiled_mma_dV;
    ThrMMA thr_mma_dV = tiled_mma_dV.get_slice(threadIdx.x);
    // dV += P^TdO
    Tensor tdVsPt = thr_mma_dV.partition_A(sPt);
    Tensor tdVsdOt = thr_mma_dV.partition_B(sdOt);
    Tensor tdVrdV_float = partition_fragment_C(tiled_mma_dV, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor tdVsdV = thr_mma_dV.partition_C(sdV);

    auto Q_TILE_MAX = size<3>(tQgQ);


    // load K, V, dK, dV tiles
    copy(gmem_tiled_copy, tKgK, tKsK);
    copy(gmem_tiled_copy, tVgV, tVsV);

    // load rL, rD from gmem to rmem
    for (int i=0; i<2; i++) {
        rL[i] = gL[thread_row + 8 * i];
    }

    CUTE_NO_UNROLL
    for (int q_tile = 0; q_tile < Q_TILE_MAX; ++q_tile) {

        // load tiles
        copy(gmem_tiled_copy, tQgQ(_,_,_,q_tile), tQsQ);
        copy(gmem_tiled_copy, tdOgdO(_,_,_,q_tile), tdOsdO);

        // compute S = QK^T

        gemm(tiled_mma, tSsQ, tSsK, tSrS_float);

        // rescale S
        for (int i=0;i< tSrS_float.size();i ++ ) {
            tSrS_float[i] *= 1.0f / sqrtf(kHeadDim);
        }

        // compute P = exp(S-l)

        // P has size blockM x blockN, partitioned by mma_S
        // gL has size (32), need to figure the L_i for each S_ij

        for (int i=0; i<2; i++) {
            for (int j=0; j< tSrS_float(make_coord(_,i),_,_).size(); j++) {
                tSrS_float(make_coord(_,i),_,_)[j] = expf(tSrS_float(make_coord(_,i),_,_)[j] - rL[i]);
            }
        }

        //convert
        constexpr int num_element = decltype(size(tSrS_float))::value;

        cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
        auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tSrS_float.data()));

        Tensor tSrP = make_tensor(make_rmem_ptr<half_t>(&frag), tSrS_float.layout());


        copy(tSrP, tSsP);

        // compute dV += p^TdO
        gemm(tiled_mma_dV, tdVsPt, tdVsdOt, tdVrdV_float);


        // compute dP = dOV^T

        // compute dS = P(dP - D)

        // compute dQ += dSK

        // compute dK += dS^TQ
    }

    // convert dV to fp16
    constexpr int num_element = decltype(size(tdVrdV_float))::value;

    cutlass::NumericArrayConverter<half_t, float, num_element> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, num_element> *>(tdVrdV_float.data()));

    Tensor tdVrdV = make_tensor(make_rmem_ptr<half_t>(&frag), tdVrdV_float.layout());

    // copy dV from rmem to smem
    copy(tdVrdV, tdVsdV);
    // copy dV from smem to gmem
//     copy(gmem_tiled_copy, tVsdV, tVgdV);
    copy(tVsdV, tVgdV);
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
    auto kernel = compute_dq_dk_dv_kernel<Flash_bwd_kernel_traits<kHeadDim, kBlockM, kBlockN, 8>>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);


    kernel<<<dimGrid, dimBlock, maxbytes>>>(q_ptr,
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