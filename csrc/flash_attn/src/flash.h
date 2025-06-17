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


template<int Headdim, bool Is_causal> void run_mha_bwd_(half_t* q,
                                                        half_t* k,
                                                        half_t* v,
                                                        half_t* o,
                                                        float* l,
                                                        float* d,
                                                        half_t* do_,
                                                        float* dq_float,
                                                        half_t* dq,
                                                        half_t* dk,
                                                        half_t* dv,
                                                        int batch_size,
                                                        int seq_len,
                                                        int num_heads,
                                                        int head_dim,
                                                        int is_causal);