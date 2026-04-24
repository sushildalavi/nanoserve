"""quality eval harness for quantized engine variants.

the engine phase (1-4) measured speed, the eval phase measures quality.
three quant paths (fp16 / int8 / int4) each get a perplexity number on a
fixed natural-text slice and an accuracy number on a small multiple-
choice completion task. the point is not to reproduce published
benchmark leaderboards — it is to show that the quant speedups (or lack
thereof) do not come at a quality cliff.

fp16 serves as the reference. the pass criterion is that int8 and int4
stay within a documented tolerance of fp16 on both metrics.
"""
