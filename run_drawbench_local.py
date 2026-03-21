#!/usr/bin/env python
"""
DrawBench image generation – local runner.

Runs a single (model, quant_mode) configuration over every prompt in the
sayakpaul/drawbench dataset and saves the images as PNG files.  Each PNG
carries its generation metadata (quant_mode, model_id, prompt, seed, …) in
embedded tEXt chunks so the file is fully self-describing.

Output layout::

    drawbench_results/<model_slug>/<quant_slug>_bs<N>/
        manifest.json          ← sorted by prompt_idx
        prompt_0000.png
        prompt_0001.png
        …

Usage
-----
    python run_drawbench_local.py \
        --model_id "black-forest-labs/FLUX.1-dev" \
        --quant_mode nvfp4

    # fp8 + compile, batch of 4 prompts per call
    python run_drawbench_local.py \
        --model_id "black-forest-labs/FLUX.1-dev" \
        --quant_mode fp8 --enable_compilation --batch_size 4
"""

import argparse
import json
import traceback as tb
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import PngImagePlugin

from benchmark import MODEL_CONFIGS, apply_compilation, setup_pipeline


def main():
    parser = argparse.ArgumentParser(description="Generate DrawBench images locally using a diffusion pipeline.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model checkpoint ID (default: black-forest-labs/FLUX.1-dev)",
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
        help="Enable torch.compile on the transformer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of prompts per pipeline call (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="drawbench_results",
        help="Root directory for outputs (default: drawbench_results)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DrawBench generation  –  local")
    print("=" * 60)
    print(f"  model_id          : {args.model_id}")
    print(f"  quant_mode        : {args.quant_mode}")
    print(f"  enable_compilation: {args.enable_compilation}")
    print(f"  batch_size        : {args.batch_size}")
    print(f"  seed              : {args.seed}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    print(f"\nLoading pipeline  model={args.model_id}  quant={args.quant_mode}  compile={args.enable_compilation}")
    pipe = setup_pipeline(args.model_id, args.quant_mode)

    if args.enable_compilation:
        print("Applying torch.compile …")
        apply_compilation(pipe)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    model_slug = args.model_id.replace("/", "_").replace("-", "_")
    quant_slug = "bf16" if args.quant_mode == "none" else args.quant_mode
    out_dir = Path(args.output_dir) / model_slug / f"{quant_slug}_bs{args.batch_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # DrawBench prompts — always loaded in dataset order (stable)
    # ------------------------------------------------------------------
    print("\nLoading sayakpaul/drawbench …")
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

    # ------------------------------------------------------------------
    # Base generation kwargs (prompt supplied per-batch, num_images fixed)
    # ------------------------------------------------------------------
    base_kwargs = {k: v for k, v in MODEL_CONFIGS[args.model_id]["call_kwargs"].items() if k != "prompt"}
    base_kwargs["num_images_per_prompt"] = 1

    # ------------------------------------------------------------------
    # Cache: skip prompts whose PNG already exists
    # ------------------------------------------------------------------
    existing_manifest: dict[int, dict] = {}
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        for entry in json.loads(manifest_path.read_text()):
            existing_manifest[entry["prompt_idx"]] = entry

    def _image_exists(idx: int) -> bool:
        p = out_dir / f"prompt_{idx:04d}.png"
        return p.exists() and p.stat().st_size > 0

    skipped = sum(1 for r in rows if _image_exists(r["idx"]))
    if skipped:
        print(f"  Skipping {skipped} already-generated images.")
    rows = [r for r in rows if not _image_exists(r["idx"])]

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------
    manifest: list[dict] = list(existing_manifest.values())
    succeeded = failed = 0

    for batch_start in range(0, len(rows), args.batch_size):
        batch = rows[batch_start : batch_start + args.batch_size]
        prompt_texts = [r["prompt"] for r in batch]

        try:
            result = pipe(
                **{**base_kwargs, "prompt": prompt_texts},
                generator=torch.manual_seed(args.seed),
            )
            for img, row in zip(result.images, batch):
                fname = f"prompt_{row['idx']:04d}.png"
                img_path = out_dir / fname

                pnginfo = PngImagePlugin.PngInfo()
                for key, val in {
                    "quant_mode": args.quant_mode,
                    "model_id": args.model_id,
                    "prompt": row["prompt"],
                    "category": row["category"],
                    "prompt_idx": str(row["idx"]),
                    "batch_size": str(args.batch_size),
                    "seed": str(args.seed),
                    "enable_compilation": str(args.enable_compilation),
                }.items():
                    pnginfo.add_text(key, val)

                img.save(img_path, format="PNG", pnginfo=pnginfo)

                manifest.append(
                    {
                        "prompt_idx": row["idx"],
                        "prompt": row["prompt"],
                        "category": row["category"],
                        "image": str(img_path),
                        "status": "success",
                    }
                )
                succeeded += 1

        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR batch [{batch[0]['idx']}…{batch[-1]['idx']}]: {exc}")
            for row in batch:
                manifest.append(
                    {
                        "prompt_idx": row["idx"],
                        "prompt": row["prompt"],
                        "category": row["category"],
                        "image": None,
                        "status": "failed",
                        "error": str(exc),
                        "traceback": tb.format_exc(),
                    }
                )
                failed += 1

        done = batch_start + len(batch)
        if done % 20 == 0 or done == len(rows):
            print(f"  {done}/{len(rows)}  ok={succeeded}  err={failed}")

    # ------------------------------------------------------------------
    # Manifest — sorted by prompt_idx for a stable, human-readable output
    # ------------------------------------------------------------------
    manifest.sort(key=lambda x: x["prompt_idx"])
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nManifest → {manifest_path}")

    print("\nDONE")
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "quant_mode": args.quant_mode,
                "enable_compilation": args.enable_compilation,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "output_dir": str(out_dir),
                "total": len(rows) + skipped,
                "skipped": skipped,
                "succeeded": succeeded,
                "failed": failed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
