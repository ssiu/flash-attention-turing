#include "flash_fwd_launch_template.h"

#include <cutlass/numeric_types.h>

using half_t = cutlass::half_t;

template<>
void run_mha_fwd_<128, true>(Flash_fwd_params &params) {
    run_mha_fwd_hdim128<true>(params);
}

