#include "flash_fwd_launch_template.h"

#include <cutlass/numeric_types.h>

using half_t = cutlass::half_t;

template<>
void run_mha_fwd_<128, true>(half_t* q,
                                               half_t* k,
                                               half_t* v,
                                               half_t* o,
                                               float* l,
                                               int batch_size,
                                               int seq_len,
                                               int num_heads,
                                               int head_dim,
                                               int is_causal) {
    run_mha_fwd_hdim128<true>(q, k, v, o, l,
                                                batch_size, seq_len, num_heads, head_dim, is_causal);
}

