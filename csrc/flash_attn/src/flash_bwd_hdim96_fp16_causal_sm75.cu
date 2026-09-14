#include "flash_bwd_launch_template.h"

#include <cutlass/numeric_types.h>

using half_t = cutlass::half_t;

template<>
void run_mha_bwd_<96, true>(Flash_bwd_params &params) {
    run_mha_bwd_hdim96<true>(params);
}
