#include "flash_fwd_launch_template.h"


template<>
void run_mha_fwd_<cutlass::half_t, 128, false>(half_t* q,
                                               half_t* k,
                                               half_t* v,
                                               half_t* o,
                                               float* l,
                                               int batch_size,
                                               int seq_len,
                                               int num_heads,
                                               int head_dim,
                                               int is_causal) {
    run_mha_fwd_hdim128<cutlass::half_t, false>(q, k, v, o, l,
                                                batch_size, seq_len, num_heads, head_dim, is_causal);
}

