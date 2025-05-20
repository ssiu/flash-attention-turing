#pragma once

#include <cutlass/numeric_types.h>
using half_t = cutlass::half_t;

template<int Headdim, bool Is_causal> void run_mha_fwd_(half_t* q,
                                                                    half_t* k,
                                                                    half_t* v,
                                                                    half_t* o,
                                                                    float* l,
                                                                    int batch_size,
                                                                    int seq_len,
                                                                    int num_heads,
                                                                    int head_dim,
                                                                    int is_causal);
