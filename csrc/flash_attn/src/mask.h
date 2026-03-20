#pragma once

#include <cute/tensor.hpp>

using namespace cute;


template <bool Is_causal>
struct Mask {

    const int max_seqlen_k, max_seqlen_q;


    __forceinline__ __device__ Mask(const int max_seqlen_q, const int max_seqlen_k)
        : max_seqlen_q(max_seqlen_q)
        , max_seqlen_k(max_seqlen_k) {
    };

    // Causal_mask: whether this particular iteration needs causal masking
    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_mask_fwd(Tensor<Engine, Layout> &tensor,
                                                    const int warp_id,
                                                    const int lane_id,
                                                    const int kBlockN,
                                                    int &causal_offset,
                                                    int &is_even_mn_offset) {
        static constexpr bool Need_masking = Causal_mask || !Is_even_MN;
        if constexpr (Need_masking) {            
            // We assume kBlockM = 128 and kBlockN = {64, 128} depending on head_dim (either 64 or 128).
            // Because we are using 8 warps, each warp is responsible for 16 rows.
            // Therefore, tSrS_float has layout ((_2,_2),_1, MMA_N),
            // since a Turing tensor core atom is 16 x 8 x 8.

            // row and col offset for tSrS_float((0, 0)), 0, 0)

            const int row_offset = (warp_id * 16) + (lane_id / 4);
            const int col_offset = (lane_id % 4) * 2;
            // int global_row_offset = m_block * kBlockM;
            // int global_col_offset = n_block * kBlockN;
            // int causal_offset = seqlen_k - seqlen_q - global_col_offset + global_row_offset;
            // int is_even_MN_offset = seqlen_k - global_col_offset;
            CUTE_UNROLL
            for (int i=0; i<2; i++) {
                for (int j=0;j<2;j++) {
                    for (int k=0; k < size<1>(tensor); k++) {
                        for (int l = 0; l < size<2>(tensor); l++) {
                            int row = row_offset + 8 * j;
                            int col = col_offset + i + 8 * l;
                            // int global_row = global_row_offset + row;
                            // int global_col = global_col_offset + col;
                            if constexpr (Causal_mask) {
                                if (col - row > causal_offset) {
                                    tensor(make_coord(i,j),k,l) = -FLT_MAX;
                                }
                            }
                            if constexpr (!Is_even_MN) {
                                if (col >= is_even_mn_offset) {
                                    tensor(make_coord(i,j),k,l) = -FLT_MAX;
                                }

                            }                            
                        }
                    }
                }
            }
            causal_offset += kBlockN;
            is_even_mn_offset += kBlockN;
        }
    };


    // Causal_mask: whether this particular iteration needs causal masking
    template <bool Causal_mask=false, bool Is_even_MN=true, 
                typename Engine_S, typename Layout_S,
                typename Engine_dP, typename Layout_dP>
    __forceinline__ __device__ void apply_mask_bwd_dq(Tensor<Engine_S, Layout_S> &tensor_S,
                                                        Tensor<Engine_dP, Layout_dP> &tensor_dP,
                                                        int warp_id,
                                                        int lane_id,
                                                        int m_block,
                                                        int n_block,
                                                        const int seqlen_q,
                                                        const int seqlen_k,
                                                        int kBlockM,
                                                        int kBlockN,
                                                        const int head_dim) {

        constexpr bool Need_masking = Causal_mask || !Is_even_MN;
        // if (cute::thread0()) { printf("Has_alibi = %d, Causal_mask=%d, Is_local=%d, Is_even_MN = %d, Need_masking = %d\n", Has_alibi, Causal_mask, Is_local, Is_even_MN, Need_masking); }
        if constexpr (Need_masking) {
            // int global_row_offset = m_block_shift * kBlockM;
            // int global_col_offset = n_block * kBlockN; 
            int row_offset = (warp_id % 2) * 16 + (lane_id / 4);
            int col_offset = (warp_id / 2) * 8 + (lane_id % 4) * 2;
            int global_row_offset = m_block * kBlockM;
            int global_col_offset = n_block * kBlockN;
            CUTE_UNROLL
            for (int i=0; i<2; i++) {
                for (int j=0;j<2;j++) {
                    for (int k=0;k<2;k++) {
                        for (int l=0;l<2;l++) {
                            int row = row_offset + 8 * j + 32 * k;
                            int col = col_offset + i + 32 * l;
                            int global_row = global_row_offset + row;
                            int global_col = global_col_offset + col;
                            if constexpr (Causal_mask) {
//                                int row = global_row_offset + row_offset + 8 * j + 32 * k;
//                                int col = global_col_offset + col_offset + i + 32 * l;
//                                if (warp_id == 0 && lane_id == 0) {
//                                    printf("warp_id = %d, lane_id = %d, i = %d, j = %d, k = %d, l = %d, row = %d, col = %d, causal_offset_local = %d\n", warp_id, lane_id, i, j, k, l, row, col, causal_offset_local);
//                                }
                                if (global_col - global_row > seqlen_k - seqlen_q) {
                                    tensor_S(make_coord(i,j),k,l) = -FLT_MAX;
                                    tensor_dP(make_coord(i,j),k,l) = 0;
                                }
                            }
                            if constexpr (!Is_even_MN) {
                                if (global_col >= seqlen_k) {
                                    tensor_S(make_coord(i,j),k,l) = -FLT_MAX;
                                    tensor_dP(make_coord(i,j),k,l) = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    // Causal_mask: whether this particular iteration needs causal masking
    template <bool Causal_mask=false, bool Is_even_MN=true, 
                typename Engine_S, typename Layout_S,
                typename Engine_dP, typename Layout_dP>
    __forceinline__ __device__ void apply_mask_bwd_dk_dv(Tensor<Engine_S, Layout_S> &tensor_S,
                                                        Tensor<Engine_dP, Layout_dP> &tensor_dP,
                                                        int warp_id,
                                                        int lane_id,
                                                        int m_block,
                                                        int n_block,
                                                        const int seqlen_q,
                                                        const int seqlen_k,
                                                        int kBlockM,
                                                        int kBlockN,
                                                        const int head_dim) {

        static constexpr bool Need_masking = Causal_mask || !Is_even_MN;
        // if (cute::thread0()) { printf("Has_alibi = %d, Causal_mask=%d, Is_local=%d, Is_even_MN = %d, Need_masking = %d\n", Has_alibi, Causal_mask, Is_local, Is_even_MN, Need_masking); }
        if constexpr (Need_masking) {
//            int global_row_offset = m_block * kBlockM;
//            int global_col_offset = n_block_shift * kBlockN;
            int row_offset = (warp_id % 2) * 16 + (lane_id / 4);
            int col_offset = (warp_id / 2) * 8 + (lane_id % 4) * 2;
            int global_row_offset = m_block * kBlockM;
            int global_col_offset = n_block * kBlockN;
            CUTE_UNROLL
            for (int i=0; i<2; i++) {
                for (int j=0;j<2;j++) {
                    for (int k=0;k<2;k++) {
                        for (int l=0;l<2;l++) {
                            int row = row_offset + 8 * j + 32 * k;
                            int col = col_offset + i + 32 * l;
                            int global_row = global_row_offset + row;
                            int global_col = global_col_offset + col; 
                            if constexpr (Causal_mask) {
//                                int row = global_row_offset + row_offset + 8 * j + 32 * k;
//                                int col = global_col_offset + col_offset + i + 32 * l;
//                                if (warp_id == 0 && lane_id == 0) {
//                                    printf("warp_id = %d, lane_id = %d, i = %d, j = %d, k = %d, l = %d, row = %d, col = %d, causal_offset_local = %d\n", warp_id, lane_id, i, j, k, l, row, col, causal_offset_local);
//                                }
                                if (global_col - global_row > seqlen_k - seqlen_q) {
                                    tensor_S(make_coord(i,j),k,l) = -FLT_MAX;
                                    tensor_dP(make_coord(i,j),k,l) = 0;
                                }          

                            }

                            if constexpr (!Is_even_MN) {
                                if (global_row >= seqlen_q) {
                                    tensor_S(make_coord(i,j),k,l) = -FLT_MAX;
                                    tensor_dP(make_coord(i,j),k,l) = 0;
                                }
                                if (global_col >= seqlen_k) {
                                    tensor_S(make_coord(i,j),k,l) = -FLT_MAX;
                                    tensor_dP(make_coord(i,j),k,l) = 0;
                                }
                                
                            }
                        }
                    }
                }
            }
        }
    }
};

