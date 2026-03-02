#!/usr/bin/env python
"""
DrawBench image generation on Modal.

Runs a single (model, quant_mode) configuration over every prompt in the
sayakpaul/drawbench dataset and saves the images as PNG files.  Each PNG
carries its generation metadata (quant_mode, model_id, prompt, seed, …) in
embedded tEXt chunks so the file is fully self-describing.

`batch_size` controls how many DrawBench prompts are batched into one
pipeline call; `num_images_per_prompt` is always 1.

Ordering guarantee
------------------
Prompts are always processed in ascending dataset-index order (0, 1, 2, …).
The output filename for row ``i`` is always ``prompt_{i:04d}.png``, which is
stable across runs because ``load_dataset`` returns rows in a fixed order.
The manifest is written sorted by ``prompt_idx``.

Usage
-----
    modal run run_drawbench_modal.py \
        --model_id "black-forest-labs/FLUX.1-dev" \
        --quant_mode nvfp4

    # fp8 + compile, batch of 4 prompts per call
    modal run run_drawbench_modal.py \
        --model_id "black-forest-labs/FLUX.1-dev" \
        --quant_mode fp8 --enable_compilation --batch_size 4
"""

from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Volume / paths
# ---------------------------------------------------------------------------
CACHE_DIR = Path("/cache")       # HF model weights
OUTPUTS_DIR = Path("/outputs")   # generated images

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("drawbench-outputs", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume, OUTPUTS_DIR: output_volume}

# ---------------------------------------------------------------------------
# Container image  (mirrors run_benchmark_modal.py exactly)
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
    .uv_pip_install(
        "transformers",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "huggingface_hub[hf_xet]",
        "datasets",
        "Pillow",
        "av",
    )
    .apt_install("git")
    .uv_pip_install(
        "diffusers @ git+https://github.com/huggingface/diffusers.git",
    )
    .add_local_python_source("benchmark")
)

app = modal.App("drawbench-generation", image=image)

# ---------------------------------------------------------------------------
# Remote function
# ---------------------------------------------------------------------------

@app.function(
    gpu="B200",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes=volumes,
)
def generate_drawbench_images(
    model_id: str = "black-forest-labs/FLUX.1-dev",
    quant_mode: str = "none",
    enable_compilation: bool = False,
    batch_size: int = 1,
    seed: int = 0,
) -> dict:
    """
    Generate one image per DrawBench prompt and save as PNG.

    Prompts are processed in batches of ``batch_size`` (each pipeline call
    receives a list of ``batch_size`` prompts; ``num_images_per_prompt`` is
    always 1).  ``torch.manual_seed(seed)`` is called at the start of each
    batch iteration so every batch receives the same RNG state.

    Output layout on the output volume::

        /outputs/<model_slug>/<quant_slug>_bs<N>/
            manifest.json          ← sorted by prompt_idx
            prompt_0000.png
            prompt_0001.png
            …

    Each PNG embeds the following metadata as tEXt chunks:

        quant_mode, model_id, prompt, category, prompt_idx,
        batch_size, seed, enable_compilation
    """
    import json
    import os
    import traceback as tb

    import torch
    from datasets import load_dataset
    from PIL import PngImagePlugin

    os.environ["HF_HOME"] = str(CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = str(CACHE_DIR)

    from benchmark import MODEL_CONFIGS, apply_compilation, setup_pipeline

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #
    print(f"Loading pipeline  model={model_id}  quant={quant_mode}  compile={enable_compilation}")
    pipe = setup_pipeline(model_id, quant_mode)

    if enable_compilation:
        print("Applying torch.compile …")
        apply_compilation(pipe)

    # ------------------------------------------------------------------ #
    # Output directory
    # ------------------------------------------------------------------ #
    model_slug = model_id.replace("/", "_").replace("-", "_")
    quant_slug = "bf16" if quant_mode == "none" else quant_mode
    out_dir = OUTPUTS_DIR / model_slug / f"{quant_slug}_bs{batch_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # DrawBench prompts — always loaded in dataset order (stable)
    # ------------------------------------------------------------------ #
    print("Loading sayakpaul/drawbench …")
    ds = load_dataset("sayakpaul/drawbench", split="train")

    prompt_col = "Prompts"
    category_col = "Category"

    rows = [
        {
            "idx": i,
            "prompt": ds[i][prompt_col],
            "category": ds[i][category_col] if category_col else "",
        }
        for i in range(len(ds))
    ]
    print(f"{len(rows)} prompts loaded")

    # ------------------------------------------------------------------ #
    # Base generation kwargs (prompt supplied per-batch, num_images fixed)
    # ------------------------------------------------------------------ #
    base_kwargs = {
        k: v
        for k, v in MODEL_CONFIGS[model_id]["call_kwargs"].items()
        if k != "prompt"
    }
    base_kwargs["num_images_per_prompt"] = 1

    # ------------------------------------------------------------------ #
    # Generation loop
    # ------------------------------------------------------------------ #
    manifest: list[dict] = []
    succeeded = failed = 0

    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start : batch_start + batch_size]
        prompt_texts = [r["prompt"] for r in batch]

        try:
            result = pipe(
                **{**base_kwargs, "prompt": prompt_texts},
                generator=torch.manual_seed(seed),
            )
            # result.images is ordered identically to prompt_texts
            for img, row in zip(result.images, batch):
                fname = f"prompt_{row['idx']:04d}.png"
                img_path = out_dir / fname

                pnginfo = PngImagePlugin.PngInfo()
                for key, val in {
                    "quant_mode": quant_mode,
                    "model_id": model_id,
                    "prompt": row["prompt"],
                    "category": row["category"],
                    "prompt_idx": str(row["idx"]),
                    "batch_size": str(batch_size),
                    "seed": str(seed),
                    "enable_compilation": str(enable_compilation),
                }.items():
                    pnginfo.add_text(key, val)

                img.save(img_path, format="PNG", pnginfo=pnginfo)

                manifest.append({
                    "prompt_idx": row["idx"],
                    "prompt": row["prompt"],
                    "category": row["category"],
                    "image": str(img_path),
                    "status": "success",
                })
                succeeded += 1

        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR batch [{batch[0]['idx']}…{batch[-1]['idx']}]: {exc}")
            for row in batch:
                manifest.append({
                    "prompt_idx": row["idx"],
                    "prompt": row["prompt"],
                    "category": row["category"],
                    "image": None,
                    "status": "failed",
                    "error": str(exc),
                    "traceback": tb.format_exc(),
                })
                failed += 1

        done = batch_start + len(batch)
        if done % 20 == 0 or done == len(rows):
            print(f"  {done}/{len(rows)}  ok={succeeded}  err={failed}")

    # ------------------------------------------------------------------ #
    # Manifest — sorted by prompt_idx for a stable, human-readable output
    # ------------------------------------------------------------------ #
    manifest.sort(key=lambda x: x["prompt_idx"])
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nManifest → {manifest_path}")

    output_volume.commit()

    return {
        "model_id": model_id,
        "quant_mode": quant_mode,
        "enable_compilation": enable_compilation,
        "batch_size": batch_size,
        "seed": seed,
        "output_dir": str(out_dir),
        "total": len(rows),
        "succeeded": succeeded,
        "failed": failed,
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
    seed: int = 0,
):
    """
    Launch DrawBench generation on a Modal B200 GPU.

    Examples
    --------
    modal run run_drawbench_modal.py \\
        --model_id "black-forest-labs/FLUX.1-dev" --quant_mode nvfp4

    modal run run_drawbench_modal.py \\
        --model_id "black-forest-labs/FLUX.1-dev" \\
        --quant_mode fp8 --enable_compilation --batch_size 4
    """
    import json

    print("=" * 60)
    print("DrawBench generation  –  Modal B200")
    print("=" * 60)
    print(f"  model_id          : {model_id}")
    print(f"  quant_mode        : {quant_mode}")
    print(f"  enable_compilation: {enable_compilation}")
    print(f"  batch_size        : {batch_size}")
    print(f"  seed              : {seed}")
    print("=" * 60)

    summary = generate_drawbench_images.remote(
        model_id=model_id,
        quant_mode=quant_mode,
        enable_compilation=enable_compilation,
        batch_size=batch_size,
        seed=seed,
    )

    print("\nDONE")
    print(json.dumps(summary, indent=2))

    # Download all generated files from the output volume to a local directory.
    # The remote out_dir is OUTPUTS_DIR / model_slug / quant_slug_bsN, so the
    # volume-relative subpath is everything after /outputs/.
    remote_out = Path(summary["output_dir"])
    volume_subpath = str(remote_out.relative_to(OUTPUTS_DIR))
    local_out = Path("drawbench_results") / volume_subpath
    local_out.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading outputs → {local_out}")
    for entry in output_volume.listdir(volume_subpath, recursive=True):
        file_data = b"".join(output_volume.read_file(entry.path))
        dest = local_out / Path(entry.path).name
        dest.write_bytes(file_data)
        print(f"  {dest}")
    print("Download complete.")
