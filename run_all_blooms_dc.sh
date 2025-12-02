#!/bin/bash
set -euo pipefail

# ---------- 공통 설정 ----------
SCRIPT_DC="decouple_train_bloom3b.py"        # 현재 디렉토리에 있다고 가정
TRAIN_FILE="input_data.txt"

NUM_GPUS=4
TOTAL_STEPS=1070           # 네가 쓰는 --total_steps
SAVE_INTERVAL=200          # 네가 쓰는 --save_interval

# 결과 기본 디렉토리 (네가 예제로 쓴 output_dir)
OUT_ROOT="$HOME/datastates-llm/deepspeed_async/LLM-Checkpoints"

DATE_TAG=$(date +%Y%m%d)

echo "========================================"
echo "  DeepSpeed-DC / Decoupled BLOOM 일괄 실행"
echo "  DATE_TAG     = ${DATE_TAG}"
echo "  TRAIN_FILE   = ${TRAIN_FILE}"
echo "  SCRIPT_DC    = ${SCRIPT_DC}"
echo "  OUT_ROOT     = ${OUT_ROOT}"
echo "  TOTAL_STEPS  = ${TOTAL_STEPS}"
echo "  SAVE_INTERVAL= ${SAVE_INTERVAL}"
echo "========================================"
echo

run_dc () {
  local MODEL_NAME="$1"   # bigscience/bloom-xxx
  local EXP_SUFFIX="$2"   # bloom056b, bloom1b1, bloom3b ...
  local CONFIG_FILE="$3"  # ds_decoupled_config.json 같은 파일

  local OUT_DIR="${OUT_ROOT}/exp_${DATE_TAG}_dc_${EXP_SUFFIX}"

  echo "----------------------------------------"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 시작 (DC): ${OUT_DIR}"
  echo "  MODEL_NAME = ${MODEL_NAME}"
  echo "  CONFIG     = ${CONFIG_FILE}"
  echo "  OUTPUT_DIR = ${OUT_DIR}"
  echo "----------------------------------------"

  mkdir -p "${OUT_DIR}"

  deepspeed --num_gpus="${NUM_GPUS}" "${SCRIPT_DC}" \
    --deepspeed_config "${CONFIG_FILE}" \
    --model_name_or_path "${MODEL_NAME}" \
    --train_file "${TRAIN_FILE}" \
    --output_dir "${OUT_DIR}" \
    --total_steps "${TOTAL_STEPS}" \
    --save_interval "${SAVE_INTERVAL}"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 완료 (DC): ${OUT_DIR}"
  echo
}

# ---------- 여기서 각 실험 돌림 ----------
# 필요하면 config 파일 이름만 바꿔서 사용하면 돼.

run_dc "bigscience/bloom-560m" "bloom056b" "ds_decoupled_config.json"
run_dc "bigscience/bloom-1b1"  "bloom1b1"  "ds_decoupled_config.json"
run_dc "bigscience/bloom-3b"   "bloom3b"   "ds_decoupled_config.json"

echo "========================================"
echo "  모든 DeepSpeed-DC / Decoupled BLOOM 실험이 완료되었습니다."
echo "========================================"