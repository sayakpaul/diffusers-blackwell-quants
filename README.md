# diffusers-blackwell-quants

Easy recipes to speed up latency of Flux, QwenImage, and LTX-2 with NVFP4 and MXFP8 on Blackwell.

> [!TIP]
> TL;DR: Using NVFP4 and MXFP8 quantization formats significantly reduces generation latency, achieving up to a ~1.7x speedup over regular baselines for video generation and ~1.4x for images. Specifically, MXFP8 consistently delivers the lowest latency for single-batch generation , while NVFP4 provides these maximum speedups at higher batch sizes (4 and 8).
