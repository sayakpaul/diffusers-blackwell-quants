#!/usr/bin/env python
"""
Compute LPIPS distance between bf16 baseline images and quantized variants.

For each prompt in DrawBench, pairs the bf16 image with the corresponding
fp8 / nvfp4 image (matched by filename) and reports per-image and aggregate
LPIPS scores.

Usage
-----
    python compute_lpips.py

    # Custom paths / output file
    python compute_lpips.py \
        --baseline_dir drawbench_results/black_forest_labs_FLUX.1_dev/bf16_bs32 \
        --compare_dirs fp8_bs32 nvfp4_bs16 \
        --output_json lpips_results.json
"""

import argparse
import json
from pathlib import Path

import lpips
import torch
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_to_tensor = transforms.Compose(
    [
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # [-1, 1] as LPIPS expects
    ]
)


def load_image_tensor(path: Path) -> torch.Tensor:
    """Load a PNG as a (1, 3, H, W) float tensor in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    return _to_tensor(img).unsqueeze(0)


def prompt_idx_from_filename(fname: str) -> int:
    """Extract the zero-padded index from 'prompt_0042.png' → 42."""
    stem = Path(fname).stem  # "prompt_0042"
    return int(stem.split("_")[1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def compute_lpips(
    baseline_dir: Path,
    compare_dirs: list[Path],
    output_json: Path,
    net: str = "alex",
) -> None:
    loss_fn = lpips.LPIPS(net=net)
    loss_fn.eval()

    # Collect baseline images keyed by prompt index.
    baseline_images = {prompt_idx_from_filename(p.name): p for p in sorted(baseline_dir.glob("prompt_*.png"))}
    if not baseline_images:
        raise FileNotFoundError(f"No prompt_*.png images found in {baseline_dir}")

    print(f"Baseline: {baseline_dir}  ({len(baseline_images)} images)")

    all_results: dict[str, dict] = {}

    for cmp_dir in compare_dirs:
        cmp_images = {prompt_idx_from_filename(p.name): p for p in sorted(cmp_dir.glob("prompt_*.png"))}
        if not cmp_images:
            print(f"  WARNING: no images found in {cmp_dir}, skipping.")
            continue

        common = sorted(set(baseline_images) & set(cmp_images))
        missing = set(baseline_images) - set(cmp_images)
        if missing:
            print(f"  WARNING: {len(missing)} baseline images have no match in {cmp_dir.name}")

        print(f"\nComparing vs {cmp_dir.name}  ({len(common)} pairs) …")

        per_image: list[dict] = []
        scores: list[float] = []

        for idx in common:
            ref_t = load_image_tensor(baseline_images[idx])
            cmp_t = load_image_tensor(cmp_images[idx])

            with torch.no_grad():
                d = loss_fn(ref_t, cmp_t).item()

            per_image.append({"prompt_idx": idx, "lpips": round(d, 6)})
            scores.append(d)

        mean_lpips = sum(scores) / len(scores)
        print(f"  mean LPIPS ({net}): {mean_lpips:.4f}  (n={len(scores)})")

        all_results[cmp_dir.name] = {
            "baseline": str(baseline_dir),
            "compared": str(cmp_dir),
            "net": net,
            "n_pairs": len(scores),
            "mean_lpips": round(mean_lpips, 6),
            "per_image": per_image,
        }

    output_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved → {output_json}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute LPIPS between bf16 baseline and quantized variants.")
    p.add_argument(
        "--baseline_dir",
        type=Path,
        default=Path("drawbench_results/black_forest_labs_FLUX.1_dev/bf16_bs32"),
    )
    p.add_argument(
        "--compare_dirs",
        type=str,
        nargs="+",
        default=["fp8_bs32", "nvfp4_bs16"],
        help="Directory names (relative to baseline_dir's parent) to compare against baseline.",
    )
    p.add_argument(
        "--output_json",
        type=Path,
        default=Path("lpips_results.json"),
    )
    p.add_argument(
        "--net",
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="Backbone for LPIPS (default: alex).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    baseline_dir = args.baseline_dir.resolve()
    parent = baseline_dir.parent
    compare_dirs = [parent / name for name in args.compare_dirs]

    compute_lpips(
        baseline_dir=baseline_dir,
        compare_dirs=compare_dirs,
        output_json=args.output_json,
        net=args.net,
    )
