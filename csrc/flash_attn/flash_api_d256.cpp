#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <climits>
#include <cstdint>
#include "flash_fwd_d256.h"
#include <c10/core/GradMode.h>

// Forward-only companion: preserve its independently qualified compiler policy.
std::vector<at::Tensor> forward(at::Tensor q, at::Tensor k, at::Tensor v,
                               double scale, bool causal) {
    TORCH_CHECK(q.dim()==4 && k.dim()==4 && v.dim()==4, "expected [B,L,H,D]");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "CUDA inputs required");
    TORCH_CHECK(q.device()==k.device() && q.device()==v.device(), "devices must match");
    TORCH_CHECK(q.scalar_type()==at::kHalf && k.scalar_type()==at::kHalf && v.scalar_type()==at::kHalf, "FP16 required");
    TORCH_CHECK(!c10::GradMode::is_enabled() ||
                (!q.requires_grad() && !k.requires_grad() && !v.requires_grad()),
                "D256 is inference only; backward is unsupported");
    for (const auto &t : {q,k,v}) for (auto n : t.sizes())
        TORCH_CHECK(n>0 && n<=INT_MAX, "positive int32 dimensions required");
    TORCH_CHECK(q.size(0)==k.size(0) && k.sizes()==v.sizes(), "batch or KV shape mismatch");
    TORCH_CHECK(q.size(3)==k.size(3), "head dimensions must match");
    TORCH_CHECK(q.size(2)%k.size(2)==0, "invalid GQA ratio");
    TORCH_CHECK(q.size(1)<=k.size(1), "append prefill requires Lq <= Lk");
    TORCH_CHECK(std::isfinite(scale) && scale>0 &&
                std::isfinite(static_cast<float>(scale)) && static_cast<float>(scale)>0,
                "scale must be finite and positive in FP32");
    const int d=q.size(3);
    TORCH_CHECK(d==256 && causal, "D256 requires causal=True");
    // Kernel offsets currently use int arithmetic. Reject overflow explicitly.
    TORCH_CHECK(q.numel()<=INT_MAX && k.numel()<=INT_MAX, "tensor too large for int32 indexing");
    const c10::cuda::CUDAGuard guard(q.device());
    const auto *prop=at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(prop->major==7 && prop->minor==5, "SM75 only");
    q=q.contiguous(); k=k.contiguous(); v=v.contiguous();
    // Contiguous views may retain a storage offset. The D256 copy atoms
    // assume 128-bit alignment, so materialize only misaligned views.
    if ((reinterpret_cast<uintptr_t>(q.data_ptr()) & 15u) != 0u) q=q.clone();
    if ((reinterpret_cast<uintptr_t>(k.data_ptr()) & 15u) != 0u) k=k.clone();
    if ((reinterpret_cast<uintptr_t>(v.data_ptr()) & 15u) != 0u) v=v.clone();
    auto o=at::empty(q.sizes(), q.options());
    auto l=at::empty({q.size(0),q.size(2),q.size(1)},q.options().dtype(at::kFloat));
    Flash_fwd_params p{};
    p.q_ptr=reinterpret_cast<half_t*>(q.data_ptr());
    p.k_ptr=reinterpret_cast<half_t*>(k.data_ptr());
    p.v_ptr=reinterpret_cast<half_t*>(v.data_ptr());
    p.o_ptr=reinterpret_cast<half_t*>(o.data_ptr());
    p.l_ptr=l.data_ptr<float>();
    p.b=q.size(0); p.seqlen_q=q.size(1); p.seqlen_k=k.size(1);
    p.h=q.size(2); p.h_k=k.size(2); p.h_h_k_ratio=p.h/p.h_k;
    p.d=d; p.softmax_scale=scale; p.is_causal=causal;
    flash_attn_d256::run_causal(p);
    return {o,l};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "SM75 forward only (contiguous conversion included)");
}
