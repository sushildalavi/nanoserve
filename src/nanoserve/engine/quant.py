"""hand-rolled INT8 weight-only quantization for nn.Linear layers.

per-row symmetric int8 quant. W is [out_features, in_features]; each row
gets a single fp16 scale, chosen so max(|W[i]|) maps to 127. weights are
stored as int8 (halving storage vs fp16), dequantized to fp16 on the fly
in the forward pass, then passed to the standard matmul.

caveat: on MPS there is no native int8 matmul kernel, so this saves
weight storage (2 GB → 1 GB for TinyLlama) but not matmul compute. the
value is that it enables larger max_batch_size within the same memory
budget, which is what shifts the regime from compute-bound to memory-
bound. see the Phase 3 section of the README for the measured effect.
"""
from __future__ import annotations

import torch
from torch import nn


class Int8LinearWeightOnly(nn.Module):
    """drop-in replacement for nn.Linear with int8 weight storage + on-the-
    fly fp16 dequantization. bias (if any) stays in fp16.
    """

    def __init__(
        self,
        weight_q: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_q", weight_q)  # [out, in] int8
        self.register_buffer("scale", scale)         # [out, 1] fp16
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> Int8LinearWeightOnly:
        W = linear.weight.data  # [out, in]
        # per-row symmetric scale. clamp to avoid divide-by-zero on dead rows.
        row_absmax = W.abs().amax(dim=-1, keepdim=True)  # [out, 1]
        scale = (row_absmax / 127.0).clamp(min=1e-8)
        W_q = (W / scale).round().clamp(-128, 127).to(torch.int8)
        bias = linear.bias.data if linear.bias is not None else None
        return cls(
            weight_q=W_q,
            scale=scale.to(torch.float16),
            bias=bias,
            in_features=linear.in_features,
            out_features=linear.out_features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # dequantize once per forward call. on MPS this materializes a fp16
        # weight tensor; on CUDA with int8 kernels we would skip this path.
        w_deq = self.weight_q.to(x.dtype) * self.scale.to(x.dtype)
        return torch.nn.functional.linear(x, w_deq, self.bias)


def quantize_model_int8_weight_only(model: nn.Module) -> tuple[int, int]:
    """replace every nn.Linear in model with Int8LinearWeightOnly in-place.
    returns (num_replaced, bytes_saved).
    """
    to_replace: list[tuple[nn.Module, str, nn.Linear]] = []
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                to_replace.append((parent, name, child))

    bytes_saved = 0
    for parent, name, lin in to_replace:
        # fp16 weight = 2 bytes * out * in. int8 = 1 byte * out * in + 2 bytes * out for scale.
        orig_bytes = 2 * lin.out_features * lin.in_features
        new_bytes = 1 * lin.out_features * lin.in_features + 2 * lin.out_features
        bytes_saved += orig_bytes - new_bytes
        q = Int8LinearWeightOnly.from_linear(lin).to(lin.weight.device)
        setattr(parent, name, q)
    return len(to_replace), bytes_saved
