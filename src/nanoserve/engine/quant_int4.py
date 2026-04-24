"""hand-rolled INT4 weight-only quantization for nn.Linear layers.

per-row symmetric signed int4 quant. each row of W [out, in] gets a single
fp16 scale chosen so max(|W[i]|) maps to 7 (signed int4 range is -8..7,
but we bias toward 7 so the symmetric clamp stays honest). two int4
values are packed into one uint8 byte (low nibble = even column, high
nibble = odd column). storage shrinks 4× vs fp16; dequant-to-fp16 runs
on every forward.

same caveat as the int8 path: on MPS there is no native int4 matmul, so
the matmul still runs in fp16. the value of int4 here is memory: weights
go from 2 GB → 512 MB (hand-rolled) for a 7B model. at TinyLlama scale
you mainly see this as a bigger effective cache budget for activations.
see the Phase 5 section of the README for measured effect.
"""
from __future__ import annotations

import torch
from torch import nn


def _pack_int4(w_int: torch.Tensor) -> torch.Tensor:
    """pack a signed int tensor with values in [-8, 7] along the last dim,
    two per byte. input shape [out, in]; output shape [out, ceil(in/2)].
    low nibble stores the even column, high nibble stores the odd column.
    odd in_features is padded with a zero column before packing.
    """
    out_f, in_f = w_int.shape
    if in_f % 2 == 1:
        pad = torch.zeros(out_f, 1, dtype=w_int.dtype, device=w_int.device)
        w_int = torch.cat([w_int, pad], dim=1)
    # map signed [-8,7] -> unsigned [0,15] via two's complement nibble
    w_u = (w_int & 0x0F).to(torch.uint8)  # [out, in_even]
    low = w_u[:, 0::2]
    high = w_u[:, 1::2]
    return (low | (high << 4)).contiguous()  # [out, ceil(in/2)] uint8


def _unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    """inverse of _pack_int4. returns int8 tensor [out, in_features] with
    values in [-8, 7].
    """
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    # interleave back into [out, ceil(in/2)*2]
    out_f = packed.shape[0]
    doubled = torch.empty(out_f, packed.shape[1] * 2, dtype=torch.int8, device=packed.device)
    doubled[:, 0::2] = low
    doubled[:, 1::2] = high
    # sign-extend: values in [8,15] represent negatives in [-8,-1]
    doubled = torch.where(doubled >= 8, doubled - 16, doubled)
    if doubled.shape[1] != in_features:
        doubled = doubled[:, :in_features]
    return doubled


class Int4LinearWeightOnly(nn.Module):
    """drop-in replacement for nn.Linear. stores weights as packed int4
    (2 per byte) + per-row fp16 scale. dequantizes to fp16 on every
    forward pass and calls nn.functional.linear.
    """

    def __init__(
        self,
        weight_packed: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_packed", weight_packed)  # [out, ceil(in/2)] uint8
        self.register_buffer("scale", scale)                  # [out, 1] fp16
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> Int4LinearWeightOnly:
        W = linear.weight.data  # [out, in]
        # per-row symmetric scale. signed int4 effective positive range is 7.
        row_absmax = W.abs().amax(dim=-1, keepdim=True)
        scale = (row_absmax / 7.0).clamp(min=1e-8)
        W_q = (W / scale).round().clamp(-8, 7).to(torch.int8)
        packed = _pack_int4(W_q)
        bias = linear.bias.data if linear.bias is not None else None
        return cls(
            weight_packed=packed,
            scale=scale.to(torch.float16),
            bias=bias,
            in_features=linear.in_features,
            out_features=linear.out_features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_int = _unpack_int4(self.weight_packed, self.in_features)
        w_deq = w_int.to(x.dtype) * self.scale.to(x.dtype)
        return torch.nn.functional.linear(x, w_deq, self.bias)


def quantize_model_int4_weight_only(model: nn.Module) -> tuple[int, int]:
    """replace every nn.Linear in model with Int4LinearWeightOnly in-place.
    returns (num_replaced, bytes_saved).
    """
    to_replace: list[tuple[nn.Module, str, nn.Linear]] = []
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                to_replace.append((parent, name, child))

    bytes_saved = 0
    for parent, name, lin in to_replace:
        orig_bytes = 2 * lin.out_features * lin.in_features
        packed_cols = (lin.in_features + 1) // 2
        new_bytes = 1 * lin.out_features * packed_cols + 2 * lin.out_features
        bytes_saved += orig_bytes - new_bytes
        q = Int4LinearWeightOnly.from_linear(lin).to(lin.weight.device)
        setattr(parent, name, q)
    return len(to_replace), bytes_saved
