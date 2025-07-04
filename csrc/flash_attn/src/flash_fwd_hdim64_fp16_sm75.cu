#include "flash_fwd_launch_template.h"

#include <cutlass/numeric_types.h>

using half_t = cutlass::half_t;

template<>
void run_mha_fwd_<64, false>(Flash_fwd_params params) {
    run_mha_fwd_hdim64<false>(params);
}

