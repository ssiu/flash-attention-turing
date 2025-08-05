#include "flash_bwd_launch_template.h"

#include <cutlass/numeric_types.h>

using half_t = cutlass::half_t;

template<>
void run_mha_bwd_<128, false>(Flash_bwd_params &params) {
    run_mha_bwd_hdim128<false>(params);
}

