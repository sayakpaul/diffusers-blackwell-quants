#!/usr/bin/env bash
# Run MX-format quantization benchmarks locally across all models, batch sizes, and quant modes.
# Compilation is always enabled with reduce-overhead mode.
#
# Usage:
#   bash run_all_benchmarks_local.sh

set -euo pipefail

MODELS=(
    "black-forest-labs/FLUX.1-dev"
    "Qwen/Qwen-Image"
    "Lightricks/LTX-2"
)

BATCH_SIZES=(1 4 8)

QUANT_MODES=(
    "none"
    "nvfp4"
    "fp8"
)

COMPILE_MODES=(
    "default"
    "reduce-overhead"
)

run() {
    local model_id="$1"
    local batch_size="$2"
    local quant_mode="$3"
    local compile_mode="$4"

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
        for quant in "${QUANT_MODES[@]}"; do
            for compile_mode in "${COMPILE_MODES[@]}"; do
                run "${model}" "${bs}" "${quant}" "${compile_mode}"
            done
        done
    done
done

echo ""
echo "All benchmarks complete."