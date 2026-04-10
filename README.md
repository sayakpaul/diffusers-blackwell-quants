# diffusers-blackwell-quants

Easy recipes to speed up latency of Flux, QwenImage, and LTX-2 with NVFP4 and MXFP8 on Blackwell.

<div align="center">
<img src="https://huggingface.co/datasets/sayakpaul/diffusers-blackwell-quants/resolve/main/plots/plot_latency_lines.png"/>
</div>

> [!NOTE]
> We demonstrate reproducible end-to-end inference speedups of up to 1.26x with MXFP8 and 1.68x with NVFP4 with diffusers and torchao on the Flux.1-Dev, QwenImage, and LTX-2 models on NVIDIA B200.  We also outline how we used selective quantization, CUDA Graphs, and LPIPS as a measure to iterate on accuracy and performance of these models.

For more information (setup, results, discussions, etc.), please refer to [our blog post ](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/).

Thanks to Claude Code for pairing 🫡

## Scripts

```bash
├── benchmark.py -- main benchmarking script which can run locally
├── compute_lpips.py -- computes lpips
├── run_all_benchmarks_local.sh -- shell script to bulk-launch runs
├── run_benchmark_local.py -- run the benchmark
└── run_drawbench_local.py -- generate images from DrawBench prompts
```

## Computing LPIPS

We provide scripts to compute LPIPS between the Bfloat16 results and the quantized results.
First, generate the images separately with each of the `quant_mode`s (including `"none"`)
with `run_drawbench_modal.py`. This will use the Drawbench dataset and the Flux.1-Dev
model. It will run on Modal.

Once all the images are generated, run `compute_lpips.py`.
