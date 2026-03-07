# diffusers-blackwell-quants

Easy recipes to speed up latency of Flux, QwenImage, and LTX-2 with NVFP4 and MXFP8 on Blackwell.

<div align="center">
<img src="https://huggingface.co/datasets/sayakpaul/diffusers-blackwell-quants/resolve/main/plots/plot_latency_lines.png"/>
</div>

> [!NOTE]
> TL;DR: NVFP4 shines at higher batch sizes (BS≥4) and for video generation (e.g., LTX-2) at any batch size, delivering up to ~38% latency reduction over bf16. For single-image inference (BS=1), MXFP8 is the better pick — NVFP4's dequantization overhead can actually hurt latency in that regime.

For more information (setup, results, discussions, etc.), please refer to our blog post (TODO).

Thanks to Claude Code for pairing 🫡

## Scripts

```bash
├── benchmark.py -- main benchmarking script which can run locally
├── compute_lpips.py -- computes lpips
├── run_all_benchmarks.sh -- shell script to bulk-launch runs
├── run_benchmark_modal.py -- run the benchmark on modal
└── run_drawbench_modal.py -- generate images from DrawBench prompts on Modal
```

## Computing LPIPS

We provide scripts to compute LPIPS between the Bfloat16 results and the quantized results.
First, generate the images separately with each of the `quant_mode`s (including `"none"`)
with `run_drawbench_modal.py`. This will use the Drawbench dataset and the Flux.1-Dev
model. It will run on Modal.

Once all the images are generated, run `compute_lpips.py`.