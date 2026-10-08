#include "flash_fwd_d256.h"
#include "flash_fwd_launch_template_d256.h"

namespace flash_attn_d256 {
void run_causal(Flash_fwd_params &params) {
    run_flash_fwd<Flash_fwd_kernel_traits<256, 64, 32, 4>, true>(params);
}
}
