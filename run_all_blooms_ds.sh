#!/bin/bash
set -euo pipefail

SCRIPT="train_bloom3b.py"
TRAIN_FILE="input_data.txt"

NUM_GPUS=4
EPOCHS=1
REQUIRED_STEPS=1070
BLOCK_SIZE=512

DATE_TAG=$(date +%Y%m%d)

echo "========================================"
echo "  DeepSpeed BLOOM 베이스라인 일괄 실행"
echo "  DATE_TAG    = ${DATE_TAG}"
echo "  TRAIN_FILE  = ${TRAIN_FILE}"
echo "  SCRIPT      = ${SCRIPT}"
echo "========================================"
echo

run_exp () {
  local MODEL_NAME="$1"
  local EXP_SUFFIX="$2"
  local CONFIG_FILE="$3"

  local OUT_DIR="exp_${DATE_TAG}_${EXP_SUFFIX}"

  echo "----------------------------------------"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 시작: ${OUT_DIR}"
  echo "  MODEL_NAME = ${MODEL_NAME}"
  echo "  CONFIG     = ${CONFIG_FILE}"
  echo "  OUTPUT_DIR = ${OUT_DIR}"
  echo "----------------------------------------"

  mkdir -p "${OUT_DIR}"

  deepspeed --num_gpus="${NUM_GPUS}" "${SCRIPT}" \
    --deepspeed_config "${CONFIG_FILE}" \
    --model_name_or_path "${MODEL_NAME}" \
    --train_file "${TRAIN_FILE}" \
    --output_dir "${OUT_DIR}" \
    --epochs "${EPOCHS}" \
    --required_steps "${REQUIRED_STEPS}" \
    --block_size "${BLOCK_SIZE}"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 완료: ${OUT_DIR}"
  echo
}

# # 0.56B – 기본 config (GPU optimizer)
# run_exp "bigscience/bloom-560m" "dsbase_bloom056b" "ds_config_zero2.json"

# # 1.1B – 기본 config
# run_exp "bigscience/bloom-1b1"  "dsbase_bloom1b1"  "ds_config_zero2.json"

# 3B – CPU offload config 사용
run_exp "bigscience/bloom-3b"   "dsbase_bloom3b"   "ds_config_zero2_3b_offload.json"

echo "========================================"
echo "  모든 DeepSpeed BLOOM 실험이 완료되었습니다."
echo "========================================"