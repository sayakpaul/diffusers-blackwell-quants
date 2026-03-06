#!/usr/bin/env python
"""
Benchmarking Script for Diffusers

Benchmarks MX-format quantization from torchao with optional torch compile
support for different diffusion models (FLUX, QwenImage, LTX-2).

Supported quant modes:
  nvfp4  – NVFP4DynamicActivationNVFP4WeightConfig
  fp8    – MXDynamicActivationMXWeightConfig (float8_e4m3fn)
  none   – bf16 baseline (no quantization)

Usage:
    python benchmark.py --model_id "black-forest-labs/FLUX.1-dev" \
        --quant_mode nvfp4 --enable_compilation
"""

import argparse
import gc
import json
import os
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import torch
import torch.utils.benchmark as benchmark

from diffusers import DiffusionPipeline


# Model configurations with their specific inference parameters
MODEL_CONFIGS = {
    "black-forest-labs/FLUX.1-dev": {
        "type": "image",
        "call_kwargs": {
            "prompt": "A cat holding a sign that says hello world",
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.5,
            "num_inference_steps": 28,
            "max_sequence_length": 512,
        },
    },
    "Qwen/Qwen-Image": {
        "type": "image",
        "call_kwargs": {
            "prompt": "A cat holding a sign that says hello world",
            "negative_prompt": " ",
            "height": 1024,
            "width": 1024,
            "true_cfg_scale": 4.0,
            "num_inference_steps": 50,
        },
    },
    "Lightricks/LTX-2": {
        "type": "video",
        "call_kwargs": {
            "prompt": (
                "A woman with long brown hair and light skin smiles at another woman with long blonde hair. "
                "The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. "
                "The camera angle is a close-up, focused on the woman with brown hair's face. "
                "The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. "
                "The scene appears to be real-life footage"
            ),
            "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
            "width": 768,
            "height": 512,
            "num_frames": 121,
            "frame_rate": 24.0,
            "num_inference_steps": 40,
            "guidance_scale": 4.0,
            "output_type": "np",
            "return_dict": False,
        },
    },
}


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    model_id: str
    quant_mode: str  # "nvfp4", "fp8", or "none"
    compilation_enabled: bool
    batch_size: int
    status: str  # "success", "failed_load", "failed_compile", "failed_warmup", "failed_benchmark", "failed_output"
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    latency_seconds: Optional[float] = None
    peak_memory_gb: Optional[float] = None
    output_path: Optional[str] = None
    inference_params: Optional[Dict[str, Any]] = field(default_factory=dict)


def flush():
    """Clean up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_run_name(model_id: str, quant_mode: str, compilation: bool, batch_size: int) -> str:
    """Generate a unique run name based on configuration."""
    model_name = model_id.replace("/", "_").replace("-", "_")
    parts = [model_name]
    parts.append("bf16" if quant_mode == "none" else quant_mode)
    parts.append("compiled" if compilation else "nocompile")
    parts.append(f"bs{batch_size}")
    return "_".join(parts)


def get_call_kwargs(config: Dict, batch_size: int) -> Dict[str, Any]:
    """Get pipeline call kwargs with batch size applied."""
    call_kwargs = config["call_kwargs"].copy()

    # Apply batch size based on model type
    if config["type"] == "image":
        call_kwargs["num_images_per_prompt"] = batch_size
    elif config["type"] == "video":
        call_kwargs["num_videos_per_prompt"] = batch_size

    return call_kwargs


def get_warmup_kwargs(call_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Create warmup kwargs with reduced inference steps."""
    warmup_kwargs = call_kwargs.copy()
    actual_steps = warmup_kwargs.get("num_inference_steps", 28)
    warmup_kwargs["num_inference_steps"] = min(5, actual_steps)
    return warmup_kwargs


def get_filter_fn(model_id: str):
    """Return a quantization filter_fn appropriate for the given model."""
    if "Qwen" in model_id:
        def filter_fn(mod, fqn):
            if not isinstance(mod, torch.nn.Linear):
                return False
            elif mod.in_features < 1024 or mod.out_features < 1024:
                return False
            # skip for accuracy reasons
            elif "embed" in fqn or "img_in" in fqn or "txt_in" in fqn:
                return False
            # skip because activation shape is small
            # M is either num_images_per_prompt, or num_images_per_prompt * 14
            elif (
                ("img_mod" in fqn)
                or ("txt_mod" in fqn)
                or ("add_q_proj" in fqn)
                or ("add_k_proj" in fqn)
                or ("add_v_proj" in fqn)
                or ("to_add_out" in fqn)
                or ("txt_mlp" in fqn)
                or ("norm_out.linear" in fqn)
            ):
                return False
            return True

    elif "LTX" in model_id:
        def filter_fn(mod, fqn):
            if not isinstance(mod, torch.nn.Linear):
                return False
            elif mod.in_features < 1024 or mod.out_features < 1024:
                return False
            # skip input/output projections for accuracy
            elif "patch_embed" in fqn or "proj_in" in fqn or "proj_out" in fqn:
                return False
            # skip text-conditioning projection (accuracy + small activations)
            elif "caption_projection" in fqn:
                return False
            # skip adaptive LayerNorm modulation (small activation: M = num_videos_per_prompt)
            elif "adaln_single" in fqn:
                return False
            # skip cross-attention context projections (text stream, small activations)
            elif (
                ("add_q_proj" in fqn)
                or ("add_k_proj" in fqn)
                or ("add_v_proj" in fqn)
                or ("to_add_out" in fqn)
            ):
                return False
            # skip output norm linear for accuracy
            elif "norm_out" in fqn:
                return False
            return True

    else:
        # FLUX
        def filter_fn(mod, fqn):
            if not isinstance(mod, torch.nn.Linear):
                return False
            elif "embed" in fqn:
                return False
            elif fqn == "norm_out.linear":
                return False
            elif fqn == "proj_out":
                return False
            elif mod.in_features < 1024 or mod.out_features < 1024:
                return False
            return True

    return filter_fn


def setup_pipeline(model_id: str, quant_mode: str, torch_dtype=torch.bfloat16):
    """Set up the diffusion pipeline with optional quantization."""
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    ).to("cuda")
    if "ltx" in model_id.lower():
        pipe.vae.enable_tiling()

    if quant_mode == "nvfp4":
        from torchao.prototype.mx_formats.inference_workflow import (
            NVFP4DynamicActivationNVFP4WeightConfig,
        )
        from torchao.quantization import quantize_

        quant_config = NVFP4DynamicActivationNVFP4WeightConfig(
            use_dynamic_per_tensor_scale=True,
            use_triton_kernel=True,
        )
        quantize_(pipe.transformer, config=quant_config, filter_fn=get_filter_fn(model_id))
        print(f"{pipe.transformer=}")

    elif quant_mode == "fp8":
        from torchao.prototype.mx_formats.inference_workflow import (
            MXDynamicActivationMXWeightConfig,
        )
        from torchao.quantization import quantize_
        from torchao.quantization.quantize_.common import KernelPreference

        quant_config = MXDynamicActivationMXWeightConfig(
            activation_dtype=torch.float8_e4m3fn,
            weight_dtype=torch.float8_e4m3fn,
            kernel_preference=KernelPreference.AUTO,
        )
        quantize_(pipe.transformer, config=quant_config, filter_fn=get_filter_fn(model_id))
        print(f"{pipe.transformer=}")

    pipe.set_progress_bar_config(disable=True)
    return pipe


def apply_compilation(pipe, fullgraph: bool = True):
    """Apply torch compile to the transformer."""
    pipe.transformer.compile_repeated_blocks(fullgraph=fullgraph)


def run_warmup(pipe, call_kwargs: Dict[str, Any], num_warmup: int, seed: int):
    """Run warmup iterations with reduced steps."""
    warmup_kwargs = get_warmup_kwargs(call_kwargs)
    for i in range(num_warmup):
        print(f"  Warmup iteration {i + 1}/{num_warmup}...")
        _ = pipe(**warmup_kwargs, generator=torch.manual_seed(seed))
        torch.cuda.synchronize()


def benchmark_fn(f, *args, **kwargs) -> float:
    """Benchmark a function using torch.utils.benchmark."""
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
        num_threads=torch.get_num_threads(),
    )
    return t0.blocked_autorange(min_run_time=5.0).mean


def run_benchmark(pipe, call_kwargs: Dict[str, Any], seed: int) -> tuple:
    """Run benchmark and return latency and peak memory."""
    flush()

    def inference():
        return pipe(**call_kwargs, generator=torch.manual_seed(seed))

    latency = benchmark_fn(inference)
    peak_memory = torch.cuda.max_memory_allocated() / 1e9

    return latency, peak_memory


def generate_final_output(pipe, call_kwargs: Dict[str, Any], seed: int):
    """Generate final output with fresh kwargs for fair visual comparison."""
    flush()
    result = pipe(**call_kwargs, generator=torch.manual_seed(seed))
    return result


def save_output(
    output, model_id: str, config: Dict, output_dir: str, quant_mode: str, compilation: bool, batch_size: int
) -> str:
    """Save the generated output (image or video)."""
    run_name = get_run_name(model_id, quant_mode, compilation, batch_size)
    os.makedirs(output_dir, exist_ok=True)

    if config["type"] == "image":
        # Handle image output
        if hasattr(output, "images"):
            image = output.images[0]
        else:
            image = output[0] if isinstance(output, (list, tuple)) else output

        output_path = os.path.join(output_dir, f"{run_name}.png")
        image.save(output_path)

    elif config["type"] == "video":
        # Handle video output (LTX-2 returns tuple: video, audio)
        from diffusers.pipelines.ltx2.export_utils import encode_video

        if isinstance(output, tuple):
            video, audio = output
        else:
            video = output
            audio = None

        assert audio is not None

        # Convert video to uint8 format
        video = (video * 255).round().astype("uint8")
        video = torch.from_numpy(video)

        output_path = os.path.join(output_dir, f"{run_name}.mp4")
        frame_rate = config["call_kwargs"].get("frame_rate", 24.0)

        if audio is not None:
            encode_video(
                video[0],
                fps=frame_rate,
                audio=audio[0].float().cpu(),
                audio_sample_rate=24000,  # LTX-2 vocoder sample rate
                output_path=output_path,
            )
        else:
            encode_video(
                video[0],
                fps=frame_rate,
                audio=None,
                audio_sample_rate=None,
                output_path=output_path,
            )

    return output_path


def save_checkpoint(result: BenchmarkResult, output_dir: str):
    """Save benchmark result to a unique checkpoint file."""
    os.makedirs(output_dir, exist_ok=True)
    run_name = get_run_name(
        result.model_id, result.quant_mode, result.compilation_enabled, result.batch_size
    )
    checkpoint_path = os.path.join(output_dir, f"{run_name}.json")
    with open(checkpoint_path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    print(f"Checkpoint saved: {checkpoint_path}")


def run_single_benchmark(model_id: str, config: Dict, args) -> BenchmarkResult:
    """Run a single benchmark with comprehensive error handling."""
    result = BenchmarkResult(
        model_id=model_id,
        quant_mode=args.quant_mode,
        compilation_enabled=args.enable_compilation,
        batch_size=args.batch_size,
        status="pending",
    )

    current_stage = "load"
    pipe = None

    try:
        # Stage 1: Load pipeline
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {model_id}")
        print(f"  Quant mode:  {args.quant_mode}")
        print(f"  Compilation: {args.enable_compilation}")
        print(f"  Batch size:  {args.batch_size}")
        print(f"{'=' * 60}")

        print("\nStage 1: Loading pipeline...")
        pipe = setup_pipeline(model_id, args.quant_mode)

        # Stage 2: Apply compilation
        current_stage = "compile"
        if args.enable_compilation:
            print("\nStage 2: Applying compilation...")
            apply_compilation(pipe)
        else:
            print("\nStage 2: Skipping compilation (disabled)")

        # Stage 3: Warmup
        current_stage = "warmup"
        call_kwargs = get_call_kwargs(config, args.batch_size)
        warmup_steps = min(5, call_kwargs.get("num_inference_steps", 28))
        print(f"\nStage 3: Running {args.num_warmup} warmup iterations (steps={warmup_steps})...")
        run_warmup(pipe, call_kwargs, args.num_warmup, args.seed)

        # Stage 4: Benchmark
        current_stage = "benchmark"
        print("\nStage 4: Benchmarking with torch.utils.benchmark...")
        latency, memory = run_benchmark(pipe, call_kwargs, args.seed)
        print(f"  Latency: {latency:.3f}s")
        print(f"  Peak memory: {memory:.2f} GB")

        # Stage 5: Generate final output
        current_stage = "output"
        print("\nStage 5: Generating final output...")
        output = generate_final_output(pipe, call_kwargs, args.seed)

        # Stage 6: Save output
        print("\nStage 6: Saving output...")
        output_path = save_output(
            output, model_id, config, args.output_dir, args.quant_mode, args.enable_compilation, args.batch_size
        )
        print(f"  Saved to: {output_path}")

        result.status = "success"
        result.latency_seconds = round(latency, 3)
        result.peak_memory_gb = round(memory, 2)
        result.output_path = output_path
        result.inference_params = call_kwargs

    except Exception as e:
        result.status = f"failed_{current_stage}"
        result.error_message = str(e)
        result.error_traceback = traceback.format_exc()
        print(f"\nERROR during {current_stage}: {e}")
        print(result.error_traceback)

    finally:
        # Cleanup to free memory for next benchmark
        if pipe is not None:
            del pipe
        flush()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MX-format quantization with torch compile for diffusion models"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model checkpoint ID to benchmark",
    )
    parser.add_argument(
        "--quant_mode",
        type=str,
        choices=["nvfp4", "fp8", "none"],
        default="none",
        help='Quantization mode: "nvfp4", "fp8", or "none" for bf16 baseline (default: none)',
    )
    parser.add_argument(
        "--enable_compilation",
        action="store_true",
        help="Enable torch compile",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of images/videos per prompt (default: 1)",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=2,
        help="Number of warmup iterations (default: 2)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save results (default: ./results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )

    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model_id]

    # Run benchmark
    result = run_single_benchmark(args.model_id, config, args)

    # Save final results
    save_checkpoint(result, args.output_dir)

    # Print summary
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"Model:       {result.model_id}")
    print(f"Status:      {result.status}")
    print(f"Quant mode:  {result.quant_mode}")
    print(f"Compilation: {result.compilation_enabled}")
    print(f"Batch size:  {result.batch_size}")

    if result.status == "success":
        print(f"Latency:     {result.latency_seconds}s")
        print(f"Peak memory: {result.peak_memory_gb} GB")
        print(f"Output:      {result.output_path}")
    else:
        print(f"Error: {result.error_message}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
