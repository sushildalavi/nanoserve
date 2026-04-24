"""MLX backend — the platform pivot Phase 3D diagnosed.

the pytorch-MPS path has no native int8/int4 matmul, so every
quantized forward pays a dequant tax to fp16. MLX is Apple's own ML
framework with a matching tensor API and native Metal kernels for int8
and int4 matmul.

this module re-implements the minimum required to benchmark
TinyLlama through MLX: model load, quantization (via `mlx.nn.quantize`),
and a streaming generate adapter that plugs into the existing
`Backend` interface so the bench harness can drive it without changes.

scope is deliberately small: serial generation only. the point of the
port is NOT to replicate the pytorch engine's scheduler — it is to
measure whether the same int4 quant path stops being a net loss when
the matmul runs in native Metal int4 instead of dequant-to-fp16.
"""
