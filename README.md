# diffusers-blackwell-quants

Easy recipes to speed up latency of Flux, QwenImage, and LTX-2 with NVFP4 and MXFP8 on Blackwell.

> [!NOTE]
> TL;DR: Using NVFP4 and MXFP8 quantization formats significantly reduces generation latency, achieving up to a ~1.7x speedup over regular baselines for video generation and ~1.4x for images. Specifically, MXFP8 consistently delivers the lowest latency for single-batch generation , while NVFP4 provides these maximum speedups at higher batch sizes (4 and 8).

Thanks to Claude Code for pairing.

## Computing LPIPS

We provide scripts to compute LPIPS between the Bfloat16 results and the quantized results.
First, generate the images separately with each of the `quant_mode`s (including `"none"`)
with `run_drawbench_modal.py`. This will use the Drawbench dataset and will use the Flux.1-Dev
model.

Once all the images are generated, run `compute_lpips.py`.