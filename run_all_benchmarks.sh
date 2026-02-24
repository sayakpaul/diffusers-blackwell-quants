#!/usr/bin/env bash
# Run MX-format quantization benchmarks across all models, batch sizes, and quant modes.
# Compilation is always enabled.
#
# Usage:
#   cd modal/
#   bash run_all_benchmarks.sh

set -euo pipefail

SCRIPT="run_benchmark_modal.py"

MODELS=(
    # "black-forest-labs/FLUX.1-dev"
    # "Qwen/Qwen-Image"
    "Lightricks/LTX-2"
)

BATCH_SIZES=(8)

run() {
    local model_id="$1"
    local batch_size="$2"
    local quant_mode="$3"   # "none", "nvfp4", or "fp8"

    echo ""
    echo "================================================================"
    echo "  model     : ${model_id}"
    echo "  batch_size: ${batch_size}"
    echo "  quant_mode: ${quant_mode}"
    echo "  compile   : enabled"
    echo "================================================================"

    modal run "${SCRIPT}" \
        --model-id "${model_id}" \
        --batch-size "${batch_size}" \
        --enable-compilation \
        --quant-mode "${quant_mode}"
}

for model in "${MODELS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        for quant in "none" "nvfp4" "fp8"; do
            run "${model}" "${bs}" "${quant}"
        done
    done
done

echo ""
echo "All benchmarks complete."
