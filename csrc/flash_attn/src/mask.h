#pragma once

#include <cute/tensor.hpp>

using namespace cute;


template <bool Is_causal>
struct Mask {

    const int max_seqlen_k, max_seqlen_q;


    __forceinline__ __device__ Mask(const int max_seqlen_q, const int max_seqlen_k)
        : max_seqlen_k(max_seqlen_q)
        , max_seqlen_q(max_seqlen_k) {
    };

    // Causal_mask: whether this particular iteration needs causal masking
    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_mask_fwd(Tensor<Engine, Layout> &tensor,
                                                    int warp_id,
                                                    int lane_id,
                                                    int kv_tile,
                                                    int KV_TILE_MASK_START,
                                                    int kBlockN) {
        static constexpr bool Need_masking = Causal_mask || !Is_even_MN;
        if constexpr (Need_masking) {            
            // We assume kBlockM = 128 and kBlockN = {64, 128} depending on head_dim (either 64 or 128).
            // Because we are using 8 warps, each warp is responsible for 16 rows.
            // Therefore, tSrS_float has layout ((_2,_2),_1, MMA_N),
            // since a Turing tensor core atom is 16 x 8 x 8.

            // row and col offset for tSrS_float((0, 0)), 0, 0)
            int row_offset = (warp_id * 16) + (lane_id / 4);
            int col_offset = (lane_id % 4) * 2 + (kv_tile - KV_TILE_MASK_START) * kBlockN;
            CUTE_UNROLL
            for (int i=0; i<2; i++) {
                for (int j=0;j<2;j++) {
                    for (int l = 0; l < size<2>(tSrS_float); l++) {
                        if constexpr (Causal_mask) {
                            int row = row_offset + 8 * j;
                            int col = col_offset + i + 8 * l;
                            if (row < col) {
                                tensor(make_coord(i,j),0,l) = -1e20;
                            }
                        }
                    }
                }
            }
        }
    };


    // // Causal_mask: whether this particular iteration needs causal masking
    // template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout>
    // __forceinline__ __device__ void apply_mask_bwd(Tensor<Engine, Layout> &tensor_,
    //                                            const int col_idx_offset_,
    //                                            const int row_idx_offset,
    //                                            const int warp_row_stride) {

    //     static constexpr bool Need_masking = Causal_mask || !Is_even_MN;
    //     // if (cute::thread0()) { printf("Has_alibi = %d, Causal_mask=%d, Is_local=%d, Is_even_MN = %d, Need_masking = %d\n", Has_alibi, Causal_mask, Is_local, Is_even_MN, Need_masking); }
    //     if constexpr (Need_masking) {
    //         // Reshape tensor_ from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
    //         Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
    //         // Do we need both row and column indices, or just column incides?
    //         static constexpr bool Col_idx_only = !(Has_alibi && !Is_causal) && !Is_local && !Causal_mask;
    //         const int lane_id = threadIdx.x % 32;
    //         const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    //         if constexpr (Col_idx_only) {
    //             #pragma unroll
    //         }
    //     }
    // };

};

