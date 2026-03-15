#!/usr/bin/env bash
# Run MX-format quantization benchmarks locally across all models, batch sizes, and quant modes.
# Compilation is always enabled with reduce-overhead mode.
#
# Usage:
#   bash run_all_benchmarks_local.sh

set -euo pipefail

MODELS=(
    "black-forest-labs/FLUX.1-dev"
    # "Qwen/Qwen-Image"
    "Lightricks/LTX-2"
)

BATCH_SIZES=(1 4 8)

run() {
    local model_id="$1"
    local batch_size="$2"
    local quant_mode="$3"   # "none", "nvfp4", or "fp8"
    local compile_mode="$4" # "default" or "reduce-overhead"

    echo ""
    echo "================================================================"
    echo "  model        : ${model_id}"
    echo "  batch_size   : ${batch_size}"
    echo "  quant_mode   : ${quant_mode}"
    echo "  compile_mode : ${compile_mode}"
    echo "================================================================"

    time python benchmark.py \
        --model_id "${model_id}" \
        --batch_size "${batch_size}" \
        --enable_compilation \
        --quant_mode "${quant_mode}" \
        --torch_compile_mode "${compile_mode}"
}

for model in "${MODELS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        for quant in "none" "nvfp4" "fp8"; do
            for compile_mode in "default" "reduce-overhead"; do
                run "${model}" "${bs}" "${quant}" "${compile_mode}"
            done
        done
    done
done

echo ""
echo "All benchmarks complete."
