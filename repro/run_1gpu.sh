#!/usr/bin/env bash
set -euo pipefail

ENV=ds0112_torch21
MODEL_NAME="${MODEL_NAME:-bigscience/bloom-7b1}"
REQUIRED_STEPS="${REQUIRED_STEPS:-20}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTDIR="${1:-$REPO/sweep_runs/smoke}"

export PATH="$HOME/miniconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TORCH_LIB="$(conda run -n "$ENV" python -c "import os,torch; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")"
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

echo "[DEBUG] REPO=$REPO"
echo "[DEBUG] MODEL_NAME=$MODEL_NAME"
echo "[DEBUG] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[DEBUG] OUTDIR=$OUTDIR"
echo "[DEBUG] TORCH_LIB=$TORCH_LIB"

rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

# single GPU면 --num_gpus 1 넣지 말고 CUDA_VISIBLE_DEVICES만 쓰는게 깔끔함
conda run -n "$ENV" deepspeed "$REPO/scripts/run_ckpt_compare_sitepkg.py" \
  --deepspeed_config "$REPO/configs/ds_config_zero1_bf16_single_a100.json" \
  --model_name_or_path "$MODEL_NAME" \
  --train_file "$REPO/input_data.txt" \
  --output_dir "$OUTDIR" \
  --required_steps "$REQUIRED_STEPS" \
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --delete_all_checkpoints_at_end 1
