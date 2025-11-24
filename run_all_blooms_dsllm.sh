#!/bin/bash
set -euo pipefail

SCRIPT="datastates_train_bloom_generic.py"
CONFIG="ds_config_zero2_datastates.json"
TRAIN_FILE="input_data.txt"

NUM_GPUS=4
EPOCHS=1
REQUIRED_STEPS=1070
BLOCK_SIZE=512

DATE_TAG=$(date +%Y%m%d)

echo "========================================"
echo "  BLOOM 체크포인트 실험 일괄 실행 시작"
echo "  DATE_TAG    = ${DATE_TAG}"
echo "  TRAIN_FILE  = ${TRAIN_FILE}"
echo "  CONFIG      = ${CONFIG}"
echo "  SCRIPT      = ${SCRIPT}"
echo "========================================"
echo

run_exp () {
  local MODEL_NAME="$1"
  local EXP_SUFFIX="$2"

  local OUT_DIR="exp_${DATE_TAG}_${EXP_SUFFIX}"

  echo "----------------------------------------"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 시작: ${OUT_DIR}"
  echo "  MODEL_NAME = ${MODEL_NAME}"
  echo "  OUTPUT_DIR = ${OUT_DIR}"
  echo "----------------------------------------"

  mkdir -p "${OUT_DIR}"

  deepspeed --num_gpus="${NUM_GPUS}" "${SCRIPT}" \
    --deepspeed_config "${CONFIG}" \
    --model_name_or_path "${MODEL_NAME}" \
    --train_file "${TRAIN_FILE}" \
    --output_dir "${OUT_DIR}" \
    --epochs "${EPOCHS}" \
    --required_steps "${REQUIRED_STEPS}" \
    --block_size "${BLOCK_SIZE}"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 완료: ${OUT_DIR}"
  echo
}

# 0.56B
run_exp "bigscience/bloom-560m" "bloom056b"

# 1.1B
run_exp "bigscience/bloom-1b1"  "bloom1b1"

# 3B
run_exp "bigscience/bloom-3b"   "bloom3b"

echo "========================================"
echo "  모든 BLOOM 실험이 완료되었습니다."
echo "========================================"