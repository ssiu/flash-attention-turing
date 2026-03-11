"""Python interface for FlashAttention Turing extension.

Call stack (dense):
    flash_attn_func
      -> FlashAttnFunc.apply
      -> FlashAttnFunc.forward
      -> _flash_attn_forward
      -> flash_attn_gpu.fwd  (pybind -> C++/CUDA)

Backward mirrors the same structure with _flash_attn_backward / flash_attn_gpu.bwd.
"""

from typing import Optional, Tuple

import flash_attn_turing as flash_attn_gpu
import torch


def maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    return flash_attn_gpu.fwd(q, k, v, causal)


def _flash_attn_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v, out, lse, dout = [maybe_contiguous(x) for x in (q, k, v, out, lse, dout)]
    return flash_attn_gpu.bwd(q, k, v, out, lse, dout, causal)


def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    cu_seqlens_q = maybe_contiguous(cu_seqlens_q)
    cu_seqlens_k = maybe_contiguous(cu_seqlens_k)
    return flash_attn_gpu.varlen_fwd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
    )


def _flash_attn_varlen_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v, out, lse, dout = [maybe_contiguous(x) for x in (q, k, v, out, lse, dout)]
    cu_seqlens_q = maybe_contiguous(cu_seqlens_q)
    cu_seqlens_k = maybe_contiguous(cu_seqlens_k)
    return flash_attn_gpu.varlen_bwd(
        q,
        k,
        v,
        out,
        lse,
        dout,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
    )


# Compatibility aliases used by local tests/benchmarks.
def fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _flash_attn_forward(q, k, v, causal)


def bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _flash_attn_backward(q, k, v, out, lse, dout, causal)


def varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
    )


def varlen_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _flash_attn_varlen_backward(
        q,
        k,
        v,
        out,
        lse,
        dout,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
    )


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool):
        out, lse = _flash_attn_forward(q, k, v, causal)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_backward(q, k, v, out, lse, dout, ctx.causal)
        return dq, dk, dv, None


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
    ):
        out, lse = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = _flash_attn_varlen_backward(
            q,
            k,
            v,
            out,
            lse,
            dout,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.causal,
        )
        return dq, dk, dv, None, None, None, None, None


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    return FlashAttnFunc.apply(q, k, v, causal)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
) -> torch.Tensor:
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
    )


__all__ = [
    "_flash_attn_forward",
    "_flash_attn_backward",
    "_flash_attn_varlen_forward",
    "_flash_attn_varlen_backward",
    "fwd",
    "bwd",
    "varlen_fwd",
    "varlen_bwd",
    "flash_attn_func",
    "flash_attn_varlen_func",
    "FlashAttnFunc",
    "FlashAttnVarlenFunc",
]
