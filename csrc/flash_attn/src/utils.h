#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>
#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>


////////////////////////////////////////////////////////////////////////////////////////////////////


template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}


////////////////////////////////////////////////////////////////////////////////////////////////////


struct MaxOp {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return fmaxf(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////


struct SumOp {
__device__ __forceinline__ float operator()(float const &x, float const &y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////



template <bool Is_even_MN=true,
            typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void masked_copy_store(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, const int warp_id, const int lane_id, const int max_MN=0) {

    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    const int row_offset = warp_id * 4 + lane_id / 8;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        int row = row_offset + m * 32;
        if (Is_even_MN || row < max_MN) {
            cute::copy(tiled_copy, S(_, m, _), D(_, m, _));
//            printf("loading row = %d, warp_id = %d, lane_id = %d\n", row, warp_id, lane_id);
        }
//        } else {
//            cute::clear(D(_, m, _));
//        }
    }
}



template <bool Is_even_MN=true,
            typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void masked_copy_read(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, const int warp_id, const int lane_id, const int max_MN=0) {

    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    const int row_offset = warp_id * 4 + lane_id / 8;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        int row = row_offset + m * 32;
        if (Is_even_MN || row < max_MN) {
            cute::copy(tiled_copy, S(_, m, _), D(_, m, _));
//            printf("loading row = %d, warp_id = %d, lane_id = %d\n", row, warp_id, lane_id);
        } else {
            cute::clear(D(_, m, _));
        }
    }
}

// template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
//           typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
//           typename Engine2, typename Layout2, typename Engine3, typename Layout3>
// __forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
//                             Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
//                             Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
//     CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
//     CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
//     CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
//     CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
//     CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
//     // There's no case where !Clear_OOB_K && Clear_OOB_MN
//     static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
//     #pragma unroll
//     for (int m = 0; m < size<1>(S); ++m) {
//         if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
//             #pragma unroll
//             for (int k = 0; k < size<2>(S); ++k) {
//                 if (Is_even_K || predicate_K(k)) {
//                     cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
//                 } else if (Clear_OOB_K) {
//                     cute::clear(D(_, m, k));
//                 }
//             }
//         } else if (Clear_OOB_MN) {
//             cute::clear(D(_, m, _));
//         }
//     }
// }
