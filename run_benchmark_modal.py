#!/usr/bin/env python
"""
Modal runner for benchmarking on B200 GPU.

Wraps benchmark.py and runs it on Modal with the same container image
used for check_compatibility.py.

Usage:
    modal run run_benchmark_modal.py \
        --model_id "black-forest-labs/FLUX.1-dev" \
        --quant_mode nvfp4 --enable_compilation
"""

from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Volume / mount setup
# ---------------------------------------------------------------------------
CACHE_DIR = Path("/cache")
RESULTS_DIR = CACHE_DIR / "benchmark_results"

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

# ---------------------------------------------------------------------------
# Container image  (mirrors check_compatibility.py exactly)
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-devel-ubuntu24.04",
        add_python="3.12",
    )
    .entrypoint([])
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .uv_pip_install(
        "torch",
        "numpy",
        extra_options="--pre --index-url https://download.pytorch.org/whl/nightly/cu129",
    )
    .uv_pip_install(
        "torchao",
        extra_options="--pre --index-url https://download.pytorch.org/whl/nightly/cu129",
    )
    .uv_pip_install("transformers", "accelerate", "sentencepiece", "protobuf", "huggingface_hub[hf_xet]", "av")
    .apt_install("git")
    .uv_pip_install(
        "diffusers @ git+https://github.com/huggingface/diffusers.git",
    )
    .add_local_python_source("benchmark")
)

app = modal.App("exotic-benchmark", image=image)

# ---------------------------------------------------------------------------
# Remote function
# ---------------------------------------------------------------------------


@app.function(
    gpu="B200",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes=volumes,
)
def run_benchmark(
    model_id: str,
    quant_mode: str = "none",
    enable_compilation: bool = False,
    batch_size: int = 1,
    num_warmup: int = 2,
    seed: int = 0,
):
    import os
    import types

    os.environ["HF_HOME"] = str(CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = str(CACHE_DIR)

    from benchmark import MODEL_CONFIGS, run_single_benchmark, save_checkpoint

    args = types.SimpleNamespace(
        quant_mode=quant_mode,
        enable_compilation=enable_compilation,
        batch_size=batch_size,
        num_warmup=num_warmup,
        seed=seed,
        output_dir=str(RESULTS_DIR),
    )

    import json
    from dataclasses import asdict

    config = MODEL_CONFIGS[model_id]
    result = run_single_benchmark(model_id, config, args)
    save_checkpoint(result, str(RESULTS_DIR))

    # Persist results to the volume.
    cache_volume.commit()

    # Read the generated image/video as raw bytes so the local caller receives
    # the actual file content without having to pull it from the volume.
    output_bytes = None
    if result.output_path is not None:
        with open(result.output_path, "rb") as f:
            output_bytes = f.read()

    return {
        "results_json": json.dumps(asdict(result), default=str),
        "output_bytes": output_bytes,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    model_id: str = "black-forest-labs/FLUX.1-dev",
    quant_mode: str = "none",
    enable_compilation: bool = False,
    batch_size: int = 1,
    num_warmup: int = 2,
    seed: int = 0,
):
    """
    Run MX-format quantization benchmark on a Modal B200 GPU.

    Examples
    --------
    modal run run_benchmark_modal.py \\
        --model_id "black-forest-labs/FLUX.1-dev" \\
        --quant_mode nvfp4 --enable_compilation

    modal run run_benchmark_modal.py \\
        --model_id "Lightricks/LTX-2" \\
        --quant_mode fp8 --enable_compilation
    """
    import json

    print("=" * 60)
    print("MX Benchmark  –  Modal B200")
    print("=" * 60)
    print(f"  model_id          : {model_id}")
    print(f"  quant_mode        : {quant_mode}")
    print(f"  enable_compilation: {enable_compilation}")
    print(f"  batch_size        : {batch_size}")
    print(f"  num_warmup        : {num_warmup}")
    print(f"  seed              : {seed}")
    print("=" * 60)

    output = run_benchmark.remote(
        model_id=model_id,
        quant_mode=quant_mode,
        enable_compilation=enable_compilation,
        batch_size=batch_size,
        num_warmup=num_warmup,
        seed=seed,
    )

    result = json.loads(output["results_json"])

    compile_str = "compiled" if enable_compilation else "eager"
    model_str = model_id.replace("/", "_").replace("-", "_")
    base_name = f"{model_str}_{quant_mode}_{compile_str}_bs{batch_size}"

    results_path = Path(f"{base_name}_results.json")
    results_path.write_text(output["results_json"])
    print(f"\nResults saved to: {results_path}")

    if output["output_bytes"] is not None:
        ext = Path(result["output_path"]).suffix
        output_path = Path(f"{base_name}_output{ext}")
        output_path.write_bytes(output["output_bytes"])
        print(f"Output saved to:  {output_path}")

    print("\nRESULT")
    print(json.dumps(result, indent=2))
