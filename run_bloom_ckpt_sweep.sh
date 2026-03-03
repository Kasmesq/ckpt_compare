#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# User-configurable defaults
# ============================================================
SCRIPT_PATH="${SCRIPT_PATH:-/home/yi/datastates-llm/LLM-Checkpoints/dsllm_p_results_new/datastates_train_bloom_generic_p_auto.py}"
DS_CONFIG="${DS_CONFIG:-/home/yi/datastates-llm/LLM-Checkpoints/dsllm_p_results_new/ds_config_zero2_local_standard.json}"
TRAIN_FILE="${TRAIN_FILE:-/home/yi/datastates-llm/LLM-Checkpoints/input_data.txt}"

NUM_GPUS="${NUM_GPUS:-4}"
EPOCHS="${EPOCHS:-1}"
REQUIRED_STEPS="${REQUIRED_STEPS:-200}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"

BASE_OUTDIR="${BASE_OUTDIR:-./sweep_runs}"
BASE_RANKLOGDIR="${BASE_RANKLOGDIR:-./sweep_rank_logs}"

KEEP_LATEST_N="${KEEP_LATEST_N:-0}"
RETENTION_CLEANUP_MODE="${RETENTION_CLEANUP_MODE:-none}"
DURABLE_MODE="${DURABLE_MODE:-auto}"

ENABLE_CKPT_BACKPRESSURE="${ENABLE_CKPT_BACKPRESSURE:-1}"
BACKPRESSURE_MODE="${BACKPRESSURE_MODE:-wait}"

AUTO_INITIAL_FREQ="${AUTO_INITIAL_FREQ:-50}"
AUTO_PROFILE_STEPS="${AUTO_PROFILE_STEPS:-25}"
AUTO_TARGET_OVERHEAD="${AUTO_TARGET_OVERHEAD:-0.05}"

ALLOW_NONEMPTY_OUTPUT_DIR="${ALLOW_NONEMPTY_OUTPUT_DIR:-0}"
LOG_DISK_USAGE_EACH_CKPT="${LOG_DISK_USAGE_EACH_CKPT:-1}"

CLEAN_BEFORE_RUN="${CLEAN_BEFORE_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# 추가 인자 전달용
EXTRA_ARGS=("$@")

mkdir -p "${BASE_OUTDIR}" "${BASE_RANKLOGDIR}"

SUMMARY_CSV="${BASE_OUTDIR}/sweep_summary.csv"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "timestamp,model_label,model_name,freq_mode,run_name,status,exit_code,output_dir,rank_log_dir,launcher_log" > "${SUMMARY_CSV}"
fi

# ============================================================
# Sweep space
# ============================================================
MODELS=(
 # "0.56|bigscience/bloom-560m|bloom_560m"
 # "1|bigscience/bloom-1b1|bloom_1b1"
  "3|bigscience/bloom-3b|bloom_3b"
)

FREQS=("101" )

# ============================================================
# Checks
# ============================================================
if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "[ERROR] SCRIPT_PATH not found: ${SCRIPT_PATH}"
  exit 1
fi

if [[ ! -f "${DS_CONFIG}" ]]; then
  echo "[ERROR] DS_CONFIG not found: ${DS_CONFIG}"
  exit 1
fi

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[ERROR] TRAIN_FILE not found: ${TRAIN_FILE}"
  exit 1
fi

echo "[INFO] SCRIPT_PATH=${SCRIPT_PATH}"
echo "[INFO] DS_CONFIG=${DS_CONFIG}"
echo "[INFO] TRAIN_FILE=${TRAIN_FILE}"
echo "[INFO] BASE_OUTDIR=${BASE_OUTDIR}"
echo "[INFO] BASE_RANKLOGDIR=${BASE_RANKLOGDIR}"
echo "[INFO] REQUIRED_STEPS=${REQUIRED_STEPS}"
echo "[INFO] BLOCK_SIZE=${BLOCK_SIZE}"
echo "[INFO] DURABLE_MODE=${DURABLE_MODE}"
echo "[INFO] KEEP_LATEST_N=${KEEP_LATEST_N}"
echo "[INFO] RETENTION_CLEANUP_MODE=${RETENTION_CLEANUP_MODE}"
echo

# ============================================================
# Run sweep
# ============================================================
for model_spec in "${MODELS[@]}"; do
  IFS='|' read -r model_label model_name model_tag <<< "${model_spec}"

  for freq in "${FREQS[@]}"; do
    run_name="${model_tag}_freq_${freq}"
    outdir="${BASE_OUTDIR}/${run_name}"
    rankdir="${BASE_RANKLOGDIR}/${run_name}"
    launcher_log="${BASE_OUTDIR}/${run_name}_launcher.log"

    echo "============================================================"
    echo "[RUN] ${run_name}"
    echo "============================================================"

    if [[ "${CLEAN_BEFORE_RUN}" == "1" ]]; then
      rm -rf "${outdir}" "${rankdir}" "${launcher_log}"
    fi

    if [[ -e "${outdir}" && "${SKIP_EXISTING}" == "1" ]]; then
      echo "[SKIP] output exists: ${outdir}"
      echo "$(date --iso-8601=seconds),${model_label},${model_name},${freq},${run_name},SKIPPED,0,${outdir},${rankdir},${launcher_log}" >> "${SUMMARY_CSV}"
      continue
    fi

    mkdir -p "${rankdir}"

    cmd=(
      deepspeed
      --num_gpus "${NUM_GPUS}"
      --enable_each_rank_log "${rankdir}"
      "${SCRIPT_PATH}"
      --deepspeed_config "${DS_CONFIG}"
      --model_name_or_path "${model_name}"
      --train_file "${TRAIN_FILE}"
      --output_dir "${outdir}"
      --experiment_name "${run_name}"
      --epochs "${EPOCHS}"
      --required_steps "${REQUIRED_STEPS}"
      --block_size "${BLOCK_SIZE}"
      --keep_latest_n "${KEEP_LATEST_N}"
      --retention_cleanup_mode "${RETENTION_CLEANUP_MODE}"
      --durable_mode "${DURABLE_MODE}"
      "${EXTRA_ARGS[@]}"
    )

    if [[ "${ALLOW_NONEMPTY_OUTPUT_DIR}" == "1" ]]; then
      cmd+=(--allow_nonempty_output_dir)
    fi

    if [[ "${LOG_DISK_USAGE_EACH_CKPT}" == "1" ]]; then
      cmd+=(--log_disk_usage_each_ckpt)
    fi

    if [[ "${ENABLE_CKPT_BACKPRESSURE}" == "1" ]]; then
      cmd+=(--enable_ckpt_backpressure --backpressure_mode "${BACKPRESSURE_MODE}")
    fi

    if [[ "${freq}" == "auto" ]]; then
      cmd+=(
        --enable_auto_freq
        --initial_freq "${AUTO_INITIAL_FREQ}"
        --auto_profile_steps "${AUTO_PROFILE_STEPS}"
        --auto_target_overhead "${AUTO_TARGET_OVERHEAD}"
      )
    else
      cmd+=(--checkpoint_interval "${freq}")
    fi

    {
      echo "[CMD-BEGIN]"
      printf '%q ' "${cmd[@]}"
      echo
      echo "[CMD-END]"
    } | tee "${launcher_log}"

    "${cmd[@]}" 2>&1 | tee -a "${launcher_log}"
    rc=${PIPESTATUS[0]}

    status="OK"
    if [[ ${rc} -ne 0 ]]; then
      status="FAIL"
    fi

    echo "$(date --iso-8601=seconds),${model_label},${model_name},${freq},${run_name},${status},${rc},${outdir},${rankdir},${launcher_log}" >> "${SUMMARY_CSV}"

    echo "[DONE] ${run_name} status=${status} rc=${rc}"
    echo
  done
done

echo "[ALL DONE] summary=${SUMMARY_CSV}"
