#include "flash_bwd_launch_template.h"

#include <cutlass/numeric_types.h>

using half_t = cutlass::half_t;

template<>
void run_mha_bwd_<128, false>(half_t* q,
                                               half_t* k,
                                               half_t* v,
                                               half_t* o,
                                               float* l,
                                               float* d,
                                               half_t* do_,
                                               half_t* dq,
                                               half_t* dk,
                                               half_t* dv,
                                               int batch_size,
                                               int seq_len,
                                               int num_heads,
                                               int head_dim,
                                               int is_causal) {
    run_mha_bwd_hdim128<false>(q, k, v, o, l, d, do_, dq, dk, dv,
                                                batch_size, seq_len, num_heads, head_dim, is_causal);
}

