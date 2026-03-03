#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datastates_train_bloom_generic_p_auto.py

Safe full-file version for DataStates + DeepSpeed + BLOOM.

핵심 정책
1) durable_mode=auto
   - single-inflight checkpoint policy
   - 훈련은 checkpoint 사이에서 계속 진행하되,
     "새 checkpoint 시작 직전"에 이전 checkpoint의 durability(wait_durable)를 보장

2) durable_mode=each_ckpt
   - 매 checkpoint 직후 즉시 wait_durable()

3) durable_mode=off
   - 완전 async override
   - 현재 3B에서는 crash 가능성이 높음

4) keep_latest_n > 0
   - old checkpoint 삭제는 마지막에만 sync로 수행
   - periodic async retention/coordinator는 이 파일에서 사용하지 않음

추가 로그
- checkpoints/metrics_log.csv
  -> checkpoint step 전용
- checkpoints/step_metrics_log.csv
  -> 모든 step 전용
- checkpoints/auto_freq_log.csv
  -> auto freq profile / overhead / freq 변경 이벤트 전용
- checkpoints/training_log.txt
  -> 텍스트 로그

추가 삭제 정책 (기본값)
- --keep_recent_n_during_run 20
  -> 런 도중 최근 20개 step checkpoint만 유지
- --delete_all_checkpoints_at_end 1
  -> 런 종료 후 step_* checkpoint 전부 삭제
"""

import argparse
import csv
import json
import math
import os
import shutil
import time
from datetime import datetime
import faulthandler

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
from datastates.llm import Checkpointing


# =============================================================================
# Global logging
# =============================================================================
LOG_FILE_PATH = None
faulthandler.enable()


def _env_rank() -> int:
    try:
        return int(os.environ.get("RANK", "0"))
    except Exception:
        return 0


def is_env_rank0() -> bool:
    return _env_rank() == 0


def r0_print(msg: str):
    if is_env_rank0():
        print(msg, flush=True)
        if LOG_FILE_PATH:
            try:
                with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{ts}] {msg}\n")
            except Exception as e:
                print(f"[Log-Error] Failed to write to log file: {e}", flush=True)


# =============================================================================
# Distributed helpers
# =============================================================================
def dist_is_ready():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def dist_barrier():
    if dist_is_ready():
        torch.distributed.barrier()


def dist_world_size():
    if dist_is_ready():
        return torch.distributed.get_world_size()
    return 1


# =============================================================================
# Auto checkpoint interval
# =============================================================================
class AutoCheckFreq:
    def __init__(self, target_overhead=0.05, initial_freq=25, profile_steps=25):
        self.target_overhead = float(target_overhead)
        self.chk_freq = int(initial_freq)
        self.profile_steps = int(profile_steps)

        self.iter_times = []
        self.avg_iter_time = 0.0
        self.is_profiling = True
        self.last_ckpt_step = 0

        self.last_cost_eval = {
            "step": None,
            "ckpt_duration_seconds": 0.0,
            "actual_interval": 0,
            "overhead_pct": 0.0,
            "required_interval": 0,
            "ckpt_equiv_iters": 0.0,
            "old_freq": self.chk_freq,
            "new_freq": self.chk_freq,
            "change_type": "none",
            "reason": "",
        }

    def snapshot(self):
        return {
            "is_profiling": self.is_profiling,
            "profiled_steps": len(self.iter_times),
            "profile_steps": self.profile_steps,
            "avg_iter_time_seconds": float(self.avg_iter_time),
            "current_freq": int(self.chk_freq),
            "last_ckpt_step": int(self.last_ckpt_step),
            "target_overhead": float(self.target_overhead),
            "last_cost_eval": dict(self.last_cost_eval),
        }

    def update_iter_time(self, duration, step=None):
        events = []

        if self.is_profiling:
            self.iter_times.append(float(duration))
            if len(self.iter_times) >= self.profile_steps:
                self.avg_iter_time = sum(self.iter_times) / len(self.iter_times)
                self.is_profiling = False

                r0_print(f"\n[AutoFreq] Profiling Done. Avg Iter Time: {self.avg_iter_time:.4f}s")
                r0_print(f"[AutoFreq] Starting with Interval: {self.chk_freq} steps\n")

                events.append({
                    "event_type": "profile_done",
                    "step": int(step) if step is not None else None,
                    "is_profiling": False,
                    "profiled_steps": len(self.iter_times),
                    "profile_steps": self.profile_steps,
                    "avg_iter_time_seconds": float(self.avg_iter_time),
                    "target_overhead": float(self.target_overhead),
                    "ckpt_duration_seconds": 0.0,
                    "actual_interval": 0,
                    "overhead_pct": 0.0,
                    "required_interval": 0,
                    "ckpt_equiv_iters": 0.0,
                    "old_freq": int(self.chk_freq),
                    "new_freq": int(self.chk_freq),
                    "change_type": "none",
                    "reason": "profiling_finished",
                })

        return events

    def should_checkpoint(self, step):
        if self.is_profiling:
            return False
        return (step - self.last_ckpt_step) >= self.chk_freq

    def update_ckpt_cost(self, ckpt_duration, step):
        ckpt_duration = float(ckpt_duration)

        if self.avg_iter_time <= 0:
            old_freq = self.chk_freq
            self.last_ckpt_step = int(step)

            info = {
                "event_type": "cost_eval",
                "step": int(step),
                "is_profiling": self.is_profiling,
                "profiled_steps": len(self.iter_times),
                "profile_steps": self.profile_steps,
                "avg_iter_time_seconds": float(self.avg_iter_time),
                "target_overhead": float(self.target_overhead),
                "ckpt_duration_seconds": ckpt_duration,
                "actual_interval": 0,
                "overhead_pct": 0.0,
                "required_interval": 0,
                "ckpt_equiv_iters": 0.0,
                "old_freq": int(old_freq),
                "new_freq": int(self.chk_freq),
                "change_type": "none",
                "reason": "avg_iter_not_ready",
            }
            self.last_cost_eval = dict(info)
            return info

        old_freq = int(self.chk_freq)
        actual_interval = max(1, int(step) - int(self.last_ckpt_step))
        current_overhead = ckpt_duration / (actual_interval * self.avg_iter_time)
        required_interval = math.ceil(ckpt_duration / (self.target_overhead * self.avg_iter_time))
        ckpt_equiv_iters = ckpt_duration / self.avg_iter_time

        r0_print(
            f"[AutoFreq] Step {step} Cost: {ckpt_duration:.2f}s | "
            f"Interval(actual)={actual_interval} | "
            f"Overhead={current_overhead*100:.2f}% | "
            f"ckpt≈{ckpt_equiv_iters:.2f} iters | "
            f"req_interval≈{required_interval}"
        )

        self.last_ckpt_step = int(step)

        change_type = "none"
        reason = "within_target"

        if current_overhead > self.target_overhead:
            inc = max(5, int(self.chk_freq * 0.2))
            self.chk_freq += inc
            change_type = "increase"
            reason = "over_target"
            r0_print(f"[AutoFreq] >>> High Overhead! Interval increase: {old_freq} -> {self.chk_freq}")

        elif current_overhead < (self.target_overhead * 0.8) and self.chk_freq > 10:
            dec = max(1, int(self.chk_freq * 0.1))
            self.chk_freq = max(10, self.chk_freq - dec)
            change_type = "decrease"
            reason = "under_target"
            r0_print(f"[AutoFreq] >>> Low Overhead! Interval decrease: {old_freq} -> {self.chk_freq}")

        info = {
            "event_type": "cost_eval",
            "step": int(step),
            "is_profiling": self.is_profiling,
            "profiled_steps": len(self.iter_times),
            "profile_steps": self.profile_steps,
            "avg_iter_time_seconds": float(self.avg_iter_time),
            "target_overhead": float(self.target_overhead),
            "ckpt_duration_seconds": ckpt_duration,
            "actual_interval": int(actual_interval),
            "overhead_pct": float(current_overhead * 100.0),
            "required_interval": int(required_interval),
            "ckpt_equiv_iters": float(ckpt_equiv_iters),
            "old_freq": int(old_freq),
            "new_freq": int(self.chk_freq),
            "change_type": change_type,
            "reason": reason,
        }
        self.last_cost_eval = dict(info)
        return info


# =============================================================================
# Utility helpers
# =============================================================================
def _parse_step_dir_name(name: str):
    if not name.startswith("step_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return None


def _list_step_dirs(ckpt_dir: str):
    step_dirs = []
    if not os.path.isdir(ckpt_dir):
        return step_dirs

    for name in os.listdir(ckpt_dir):
        step_no = _parse_step_dir_name(name)
        if step_no is None:
            continue
        path = os.path.join(ckpt_dir, name)
        if os.path.isdir(path):
            step_dirs.append((step_no, path))

    step_dirs.sort(key=lambda x: x[0])
    return step_dirs


def cleanup_old_steps_sync(ckpt_dir: str, keep_n: int):
    t0 = time.time()

    if keep_n <= 0 or not os.path.isdir(ckpt_dir):
        return {
            "deleted_count": 0,
            "deleted_steps": [],
            "duration_seconds": time.time() - t0,
        }

    step_dirs = _list_step_dirs(ckpt_dir)
    victims = step_dirs[:-keep_n]

    deleted_steps = []
    for step_no, path in victims:
        try:
            r0_print(f"[Retention-Run] deleting old checkpoint step={step_no} path={path}")
            shutil.rmtree(path)
            deleted_steps.append(step_no)
        except FileNotFoundError:
            pass

    return {
        "deleted_count": len(deleted_steps),
        "deleted_steps": deleted_steps,
        "duration_seconds": time.time() - t0,
    }


def delete_all_steps_sync(ckpt_dir: str):
    t0 = time.time()

    if not os.path.isdir(ckpt_dir):
        return {
            "deleted_count": 0,
            "deleted_steps": [],
            "duration_seconds": time.time() - t0,
        }

    step_dirs = _list_step_dirs(ckpt_dir)

    deleted_steps = []
    for step_no, path in step_dirs:
        try:
            r0_print(f"[Retention-Final] deleting checkpoint step={step_no} path={path}")
            shutil.rmtree(path)
            deleted_steps.append(step_no)
        except FileNotFoundError:
            pass

    return {
        "deleted_count": len(deleted_steps),
        "deleted_steps": deleted_steps,
        "duration_seconds": time.time() - t0,
    }


def _append_reason(existing: str, new_reason: str) -> str:
    if not new_reason:
        return existing
    if not existing:
        return new_reason
    parts = existing.split("|")
    if new_reason in parts:
        return existing
    return existing + "|" + new_reason


def move_state_to_cpu(state, label="state"):
    tensor_count, total_numel = 0, 0

    def _to_cpu(obj):
        nonlocal tensor_count, total_numel
        if isinstance(obj, torch.Tensor):
            tensor_count += 1
            total_numel += obj.numel()
            return obj.detach().cpu()
        elif isinstance(obj, dict):
            return {k: _to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            typ = type(obj)
            return typ(_to_cpu(v) for v in obj)
        else:
            return obj

    cpu_state = _to_cpu(state)
    return cpu_state, tensor_count, total_numel


# ----------------------------------------------------
# BLOOM tied-weight dedup helpers
# ----------------------------------------------------
TIED_WEIGHT_CANDIDATE_PAIRS = (
    ("transformer.word_embeddings.weight", "lm_head.weight"),
    ("module.transformer.word_embeddings.weight", "module.lm_head.weight"),
)


def _new_mapping_like(mapping):
    try:
        return mapping.__class__()
    except Exception:
        return {}


def _tensor_storage_ptr(t: torch.Tensor) -> int:
    getters = (
        lambda x: x.untyped_storage().data_ptr(),
        lambda x: x.storage().data_ptr(),
        lambda x: x.data_ptr(),
    )
    for getter in getters:
        try:
            return int(getter(t))
        except Exception:
            pass
    return -1


def _find_tied_weight_pair(state):
    for embed_key, head_key in TIED_WEIGHT_CANDIDATE_PAIRS:
        if embed_key not in state or head_key not in state:
            continue

        emb = state[embed_key]
        head = state[head_key]

        if not isinstance(emb, torch.Tensor) or not isinstance(head, torch.Tensor):
            continue

        try:
            same_storage = _tensor_storage_ptr(emb) == _tensor_storage_ptr(head)
        except Exception:
            same_storage = False

        if same_storage and emb.shape == head.shape and emb.dtype == head.dtype:
            return embed_key, head_key, emb

    return None, None, None


def maybe_strip_tied_output_head(state):
    embed_key, head_key, emb = _find_tied_weight_pair(state)
    if embed_key is None:
        return state, False

    pruned = _new_mapping_like(state)
    for k, v in state.items():
        if k != head_key:
            pruned[k] = v

    r0_print(
        f"[Model-CKPT] tied weights detected; dropping duplicate key '{head_key}' "
        f"and keeping '{embed_key}' only | numel={emb.numel()} dtype={emb.dtype}"
    )
    return pruned, True


def maybe_restore_tied_output_head(state):
    for embed_key, head_key in TIED_WEIGHT_CANDIDATE_PAIRS:
        if embed_key in state and head_key not in state:
            emb = state[embed_key]
            if not isinstance(emb, torch.Tensor):
                continue

            restored = _new_mapping_like(state)
            restored.update(state)
            restored[head_key] = emb

            r0_print(
                f"[Model-LOAD] restored missing tied key '{head_key}' from '{embed_key}' "
                f"| numel={emb.numel()} dtype={emb.dtype}"
            )
            return restored

    return state


# ----------------------------------------------------
# Checkpoint engine wrappers
# ----------------------------------------------------
def checkpointing_wait_background(checkpointing):
    if hasattr(checkpointing, "wait") and callable(getattr(checkpointing, "wait")):
        return checkpointing.wait()
    return 0.0


def checkpointing_wait_durable(checkpointing):
    if hasattr(checkpointing, "wait_durable") and callable(getattr(checkpointing, "wait_durable")):
        return checkpointing.wait_durable()
    if hasattr(checkpointing, "ckpt_engine") and hasattr(checkpointing.ckpt_engine, "wait_durable"):
        return checkpointing.ckpt_engine.wait_durable()
    raise RuntimeError("[Durability] wait_durable() missing")


def checkpointing_has_wait_durable(checkpointing) -> bool:
    if hasattr(checkpointing, "wait_durable") and callable(getattr(checkpointing, "wait_durable")):
        return True
    if hasattr(checkpointing, "ckpt_engine") and hasattr(checkpointing.ckpt_engine, "has_wait_durable"):
        try:
            return bool(checkpointing.ckpt_engine.has_wait_durable())
        except Exception:
            return hasattr(checkpointing.ckpt_engine, "wait_durable")
    if hasattr(checkpointing, "ckpt_engine") and hasattr(checkpointing.ckpt_engine, "wait_durable"):
        return True
    return False


def checkpointing_close_safely(checkpointing):
    if checkpointing is None:
        return

    if hasattr(checkpointing, "close") and callable(getattr(checkpointing, "close")):
        checkpointing.close()
        return

    if hasattr(checkpointing, "ckpt_engine") and hasattr(checkpointing.ckpt_engine, "close"):
        checkpointing.ckpt_engine.close()
        return


# =============================================================================
# Dataset / CSV / JSON
# =============================================================================
def load_dataset_stream(file_path, tokenizer, block_size=512):
    blocks, buffer = [], []

    if not os.path.exists(file_path):
        return [torch.randint(0, 1000, (block_size,), dtype=torch.long) for _ in range(100)]

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            tokens = tokenizer(line, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
            buffer.append(tokens)

            while sum(b.numel() for b in buffer) >= block_size:
                concat = torch.cat(buffer)
                blocks.append(concat[:block_size])
                buffer = [concat[block_size:]] if concat.numel() > block_size else []

    if buffer:
        concat = torch.cat(buffer)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        if concat.numel() < block_size:
            concat = torch.nn.functional.pad(
                concat,
                (0, block_size - concat.numel()),
                value=pad_id,
            )
        blocks.append(concat)

    return blocks


def append_csv_row(csv_path: str, row):
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def init_checkpoint_csv_log(csv_path: str):
    if os.path.exists(csv_path):
        return

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "experiment_name",
            "epoch",
            "step",
            "loss",
            "rank",
            "total_ckpt_seconds",
            "enqueue_seconds",
            "flush_barrier_seconds",
            "durable_local_seconds",
            "durable_window_seconds",
            "cleanup_seconds",
            "cleanup_deleted_count",
            "model_gpu2cpu_seconds",
            "model_cpu2disk_seconds",
            "optim_gpu2cpu_seconds",
            "optim_cpu2disk_seconds",
            "model_tensor_elements",
            "optim_tensor_elements",
            "model_logical_bytes",
            "optim_logical_bytes",
            "model_throughput_GBps",
            "optim_throughput_GBps",
            "backward_seconds",
            "step_seconds",
            "host_cache_capacity_bytes",
            "host_cache_used_bytes_t1",
            "host_cache_used_bytes_t2",
            "pending_flush_bytes_t1",
            "pending_flush_bytes_t2",
            "flush_queue_depth_t1",
            "flush_queue_depth_t2",
            "process_rss_bytes_t1",
            "process_rss_bytes_t2",
            "process_vmlck_bytes_t1",
            "process_vmlck_bytes_t2",
        ])


def init_step_csv_log(csv_path: str):
    if os.path.exists(csv_path):
        return

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "experiment_name",
            "epoch",
            "step",
            "rank",
            "loss",
            "compute_iter_seconds",
            "step_total_wall_seconds",
            "post_compute_overhead_seconds",
            "backward_seconds",
            "optimizer_step_seconds",
            "checkpoint_requested_initial",
            "checkpoint_reason",
            "checkpoint_requested_final",
            "checkpoint_executed",
            "checkpoint_skipped_by_backpressure",
            "checkpoint_tag",
            "durable_mode_effective",
            "pending_durable_tag_before",
            "pending_durable_tag_after",
            "wait_prev_durable_triggered",
            "wait_prev_durable_local_seconds",
            "wait_prev_durable_window_seconds",
            "wait_prev_durable_bp_wait_seconds",
            "pre_ckpt_backpressure_wait_seconds",
            "ckpt_total_seconds",
            "ckpt_enqueue_seconds",
            "ckpt_flush_barrier_seconds",
            "ckpt_wait_background_seconds",
            "ckpt_durable_local_seconds",
            "ckpt_durable_window_seconds",
            "cleanup_triggered_this_step",
            "cleanup_deleted_count_this_step",
            "cleanup_seconds_this_step",
            "cleanup_reason_this_step",
            "host_cache_capacity_bytes",
            "host_cache_used_bytes_t1",
            "host_cache_used_bytes_t2",
            "pending_flush_bytes_t1",
            "pending_flush_bytes_t2",
            "flush_queue_depth_t1",
            "flush_queue_depth_t2",
            "process_rss_bytes_t1",
            "process_rss_bytes_t2",
            "process_vmlck_bytes_t1",
            "process_vmlck_bytes_t2",
            "auto_enabled",
            "auto_is_profiling",
            "auto_profiled_steps",
            "auto_profile_steps",
            "auto_target_overhead",
            "auto_avg_iter_time_seconds",
            "auto_current_freq",
            "auto_last_ckpt_step",
            "auto_last_eval_step",
            "auto_last_eval_ckpt_duration_seconds",
            "auto_last_eval_actual_interval",
            "auto_last_eval_overhead_pct",
            "auto_last_eval_required_interval",
            "auto_last_eval_ckpt_equiv_iters",
            "auto_last_eval_old_freq",
            "auto_last_eval_new_freq",
            "auto_last_eval_change_type",
        ])


def init_auto_freq_csv_log(csv_path: str):
    if os.path.exists(csv_path):
        return

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "experiment_name",
            "event_type",
            "step",
            "is_profiling",
            "profiled_steps",
            "profile_steps",
            "avg_iter_time_seconds",
            "target_overhead",
            "ckpt_duration_seconds",
            "actual_interval",
            "overhead_pct",
            "required_interval",
            "ckpt_equiv_iters",
            "old_freq",
            "new_freq",
            "change_type",
            "reason",
        ])


def write_json(path: str, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# =============================================================================
# Instrumentation helpers
# =============================================================================
def _safe_int(value, default=0):
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if hasattr(value, "item"):
        try:
            return int(value.item())
        except Exception:
            pass
    try:
        return int(value)
    except Exception:
        return default


def _read_proc_status_memory_bytes():
    rss_bytes = 0
    vmlck_bytes = 0
    status_path = "/proc/self/status"

    if not os.path.exists(status_path):
        return rss_bytes, vmlck_bytes

    try:
        with open(status_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        rss_bytes = int(parts[1]) * 1024
                elif line.startswith("VmLck:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        vmlck_bytes = int(parts[1]) * 1024
    except Exception:
        pass

    return rss_bytes, vmlck_bytes


def _get_nested_attr(obj, path):
    cur = obj
    for name in path:
        if cur is None or not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
        if callable(cur):
            try:
                cur = cur()
            except TypeError:
                pass
            except Exception:
                return None
    return cur


def _extract_queue_depth(value):
    if value is None:
        return 0

    if hasattr(value, "qsize") and callable(value.qsize):
        try:
            return int(value.qsize())
        except Exception:
            pass

    try:
        return int(len(value))
    except Exception:
        return _safe_int(value, 0)


def _normalize_ckpt_stats(snapshot, runtime_config=None):
    runtime_config = runtime_config or {}
    default_capacity_bytes = _safe_int(runtime_config.get("host_cache_size", 0)) * (1024 ** 3)

    snapshot = snapshot or {}
    return {
        "host_cache_capacity_bytes": _safe_int(snapshot.get("host_cache_capacity_bytes"), default_capacity_bytes),
        "host_cache_used_bytes": _safe_int(snapshot.get("host_cache_used_bytes"), 0),
        "pending_flush_bytes": _safe_int(snapshot.get("pending_flush_bytes"), 0),
        "flush_queue_depth": _safe_int(snapshot.get("flush_queue_depth"), 0),
    }


def _build_ckpt_engine_stats_snapshot(checkpointing, runtime_config):
    def _resolve_scalar(candidate_paths, default=0):
        for path in candidate_paths:
            value = _get_nested_attr(checkpointing, path)
            if value is not None:
                return _safe_int(value, default)
        return default

    def _resolve_queue_depth(candidate_paths, default=0):
        for path in candidate_paths:
            value = _get_nested_attr(checkpointing, path)
            if value is not None:
                return _extract_queue_depth(value)
        return default

    def _stats_snapshot():
        default_capacity_bytes = _safe_int(runtime_config.get("host_cache_size", 0)) * (1024 ** 3)
        return {
            "host_cache_capacity_bytes": _resolve_scalar([
                ("host_cache_capacity_bytes",),
                ("host_cache_capacity",),
                ("cache_capacity_bytes",),
                ("_host_cache_capacity_bytes",),
                ("host_cache", "capacity_bytes"),
                ("host_cache", "total_bytes"),
                ("cache", "capacity_bytes"),
                ("cache", "total_bytes"),
            ], default_capacity_bytes),
            "host_cache_used_bytes": _resolve_scalar([
                ("host_cache_used_bytes",),
                ("host_cache_used",),
                ("cache_used_bytes",),
                ("_host_cache_used_bytes",),
                ("host_cache", "used_bytes"),
                ("host_cache", "current_bytes"),
                ("cache", "used_bytes"),
                ("cache", "current_bytes"),
            ], 0),
            "pending_flush_bytes": _resolve_scalar([
                ("pending_flush_bytes",),
                ("flush_pending_bytes",),
                ("_pending_flush_bytes",),
                ("flush_manager", "pending_flush_bytes"),
                ("flush_state", "pending_flush_bytes"),
            ], 0),
            "flush_queue_depth": _resolve_scalar([
                ("flush_queue_depth",),
                ("_flush_queue_depth",),
                ("flush_manager", "flush_queue_depth"),
            ], _resolve_queue_depth([
                ("flush_queue",),
                ("_flush_queue",),
                ("queue",),
                ("flush_manager", "flush_queue"),
                ("flush_manager", "queue"),
            ], 0)),
        }

    return _stats_snapshot


def ensure_ckpt_engine_stats_snapshot(checkpointing, runtime_config):
    fallback_stats_snapshot = _build_ckpt_engine_stats_snapshot(checkpointing, runtime_config)

    if hasattr(checkpointing, "stats_snapshot") and callable(getattr(checkpointing, "stats_snapshot")):
        original_stats_snapshot = checkpointing.stats_snapshot

        def _wrapped_stats_snapshot():
            try:
                return _normalize_ckpt_stats(original_stats_snapshot(), runtime_config)
            except Exception:
                return _normalize_ckpt_stats(fallback_stats_snapshot(), runtime_config)

        checkpointing.stats_snapshot = _wrapped_stats_snapshot
    else:
        def _wrapped_stats_snapshot():
            return _normalize_ckpt_stats(fallback_stats_snapshot(), runtime_config)

        checkpointing.stats_snapshot = _wrapped_stats_snapshot


# =============================================================================
# Output-dir / disk / backpressure
# =============================================================================
def _human_bytes(n: int) -> str:
    x = float(max(0, int(n)))
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f}{u}"
        x /= 1024.0
    return f"{n}B"


def _nearest_existing_path(path: str) -> str:
    cur = os.path.abspath(path)
    while not os.path.exists(cur):
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(".")
        cur = parent
    return cur


def _preview_dir_entries(path: str, limit: int = 8) -> str:
    try:
        names = sorted(os.listdir(path))
    except Exception as e:
        return f"<unreadable: {e}>"

    if not names:
        return "<empty>"

    if len(names) <= limit:
        return ", ".join(names)

    return f"{', '.join(names[:limit])}, ... (+{len(names) - limit} more)"


def enforce_output_dir_guard(output_dir: str, resume_from: str = None, allow_nonempty: bool = False):
    if not is_env_rank0():
        return

    if allow_nonempty:
        return
    if resume_from:
        return
    if not os.path.isdir(output_dir):
        return

    entries = [x for x in os.listdir(output_dir) if x not in (".DS_Store",)]
    if entries:
        preview = _preview_dir_entries(output_dir)
        raise RuntimeError(
            "[OutputDir-Guard] Refusing to reuse a non-empty output_dir without "
            "--resume_from or --allow_nonempty_output_dir.\n"
            f"output_dir={os.path.abspath(output_dir)}\n"
            f"entries={preview}"
        )


def _read_disk_usage(path: str):
    target = _nearest_existing_path(path)
    du = shutil.disk_usage(target)
    st = os.statvfs(target)

    inode_total = int(st.f_files)
    inode_free = int(st.f_ffree)
    inode_used = max(0, inode_total - inode_free)

    return {
        "path": target,
        "total_bytes": int(du.total),
        "used_bytes": int(du.used),
        "free_bytes": int(du.free),
        "use_pct": (float(du.used) / float(du.total) * 100.0) if du.total else 0.0,
        "inode_total": inode_total,
        "inode_used": inode_used,
        "inode_free": inode_free,
        "inode_use_pct": (float(inode_used) / float(inode_total) * 100.0) if inode_total else 0.0,
    }


def log_disk_usage(path: str, prefix: str = ""):
    info = _read_disk_usage(path)
    r0_print(
        f"{prefix}[DISK] "
        f"path={info['path']} | "
        f"free={info['free_bytes']} ({_human_bytes(info['free_bytes'])}) | "
        f"used={info['used_bytes']} ({_human_bytes(info['used_bytes'])}) | "
        f"total={info['total_bytes']} ({_human_bytes(info['total_bytes'])}) | "
        f"use={info['use_pct']:.2f}% | "
        f"inode_free={info['inode_free']} | "
        f"inode_used={info['inode_used']} | "
        f"inode_use={info['inode_use_pct']:.2f}%"
    )


def local_backpressure_snapshot(checkpointing):
    try:
        stats = checkpointing.stats_snapshot() or {}
    except Exception:
        stats = {}

    snap = {
        "pending_flush_bytes": _safe_int(stats.get("pending_flush_bytes"), 0),
        "flush_queue_depth": _safe_int(stats.get("flush_queue_depth"), 0),
    }
    snap["active"] = bool(
        snap["pending_flush_bytes"] > 0 or snap["flush_queue_depth"] > 0
    )
    return snap


def global_backpressure_active(ds_engine, checkpointing):
    local_snap = local_backpressure_snapshot(checkpointing)
    active = local_snap["active"]

    if dist_is_ready():
        flag = torch.tensor(
            [1 if active else 0],
            dtype=torch.int32,
            device=ds_engine.device,
        )
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        active = bool(flag.item())

    return active, local_snap


def wait_for_backpressure_clear(
    ds_engine,
    checkpointing,
    poll_s: float = 0.05,
    timeout_s: float = None,
    label: str = "",
):
    t0 = time.time()
    last_log_t = 0.0

    while True:
        active, local_snap = global_backpressure_active(ds_engine, checkpointing)
        if not active:
            waited = time.time() - t0
            if ds_engine.global_rank == 0:
                r0_print(
                    f"[Backpressure] cleared label={label} waited={waited:.3f}s | "
                    f"local_pending_flush={local_snap['pending_flush_bytes']} "
                    f"({_human_bytes(local_snap['pending_flush_bytes'])}) | "
                    f"local_q={local_snap['flush_queue_depth']}"
                )
            return waited

        now = time.time()
        if ds_engine.global_rank == 0 and (now - last_log_t) >= 1.0:
            r0_print(
                f"[Backpressure] active label={label} | "
                f"local_pending_flush={local_snap['pending_flush_bytes']} "
                f"({_human_bytes(local_snap['pending_flush_bytes'])}) | "
                f"local_q={local_snap['flush_queue_depth']}"
            )
            last_log_t = now

        if timeout_s is not None and timeout_s > 0 and (now - t0) >= timeout_s:
            if ds_engine.global_rank == 0:
                r0_print(
                    f"[Backpressure-Warn] timeout label={label} waited={now - t0:.3f}s | "
                    f"local_pending_flush={local_snap['pending_flush_bytes']} "
                    f"({_human_bytes(local_snap['pending_flush_bytes'])}) | "
                    f"local_q={local_snap['flush_queue_depth']}"
                )
            return now - t0

        time.sleep(max(0.01, poll_s))


# =============================================================================
# Main
# =============================================================================
def main():
    global LOG_FILE_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed_config", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="bigscience/bloom-3b")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./bloom-finetuned")
    parser.add_argument("--experiment_name", type=str, default=None)

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--required_steps", type=int, default=1070)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--checkpoint_interval", type=int, default=200)

    parser.add_argument("--allow_nonempty_output_dir", action="store_true")
    parser.add_argument("--log_disk_usage_each_ckpt", action="store_true")

    parser.add_argument("--enable_ckpt_backpressure", action="store_true")
    parser.add_argument("--backpressure_mode", type=str, default="skip", choices=["skip", "wait"])
    parser.add_argument("--backpressure_poll_s", type=float, default=0.05)
    parser.add_argument("--backpressure_timeout_s", type=float, default=0.0)

    parser.add_argument("--enable_auto_freq", action="store_true")
    parser.add_argument("--initial_freq", type=int, default=50)
    parser.add_argument("--auto_profile_steps", type=int, default=25)
    parser.add_argument("--auto_target_overhead", type=float, default=0.05)

    parser.add_argument("--keep_latest_n", type=int, default=0)

    parser.add_argument(
        "--retention_cleanup_mode",
        type=str,
        default="end",
        choices=["none", "end", "periodic_safe"],
        help=(
            "이 파일에서는 periodic_safe를 end로 강등한다. "
            "none이면 삭제 안 함, end면 마지막에만 old checkpoint 삭제"
        ),
    )

    parser.add_argument(
        "--durable_mode",
        type=str,
        default="auto",
        choices=["auto", "off", "before_next_ckpt", "each_ckpt"],
        help=(
            "auto: single-inflight(before_next_ckpt). "
            "off: 완전 async override(위험). "
            "before_next_ckpt: 새 checkpoint 시작 직전에 이전 checkpoint wait_durable(). "
            "each_ckpt: 매 checkpoint 직후 wait_durable()."
        ),
    )

    # 네 요청대로 기본값 설정
    parser.add_argument(
        "--keep_recent_n_during_run",
        type=int,
        default=20,
        help="런 도중 안전 지점마다 최근 N개 step checkpoint만 유지. 기본값=20",
    )
    parser.add_argument(
        "--delete_all_checkpoints_at_end",
        type=int,
        default=1,
        choices=[0, 1],
        help="1이면 런 종료 후 step_* checkpoint 전부 삭제. 기본값=1",
    )

    parser.add_argument("--retention_final_wait_timeout_s", type=float, default=120.0)
    args = parser.parse_args()

    checkpointing = None

    if args.experiment_name is None or len(str(args.experiment_name).strip()) == 0:
        args.experiment_name = os.path.basename(os.path.abspath(args.output_dir))

    if args.retention_cleanup_mode == "periodic_safe":
        r0_print(
            "[WARN] retention_cleanup_mode=periodic_safe is disabled in this safe full-file version; "
            "coercing to retention_cleanup_mode=end"
        )
        args.retention_cleanup_mode = "end"

    enforce_output_dir_guard(
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        allow_nonempty=args.allow_nonempty_output_dir,
    )

    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)

    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_ckpt_cfg = ds_config.get("datastates_ckpt", {})
    ds_ckpt_cfg.setdefault("host_cache_size", 32)
    ds_ckpt_cfg.setdefault("parser_threads", 8)
    ds_config["datastates_ckpt"] = ds_ckpt_cfg

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    metrics_csv = os.path.join(ckpt_dir, "metrics_log.csv")
    step_metrics_csv = os.path.join(ckpt_dir, "step_metrics_log.csv")
    auto_freq_csv = os.path.join(ckpt_dir, "auto_freq_log.csv")
    run_config_json = os.path.join(ckpt_dir, "run_config.json")

    if is_env_rank0():
        import datastates
        import datastates.ckpt.src as ckpt_src

        r0_print(f"[IMPORT] datastates={getattr(datastates, '__file__', None)}")
        r0_print(f"[IMPORT] datastates.ckpt.src={getattr(ckpt_src, '__file__', None)}")

        os.makedirs(ckpt_dir, exist_ok=True)
        LOG_FILE_PATH = os.path.join(ckpt_dir, "training_log.txt")

        init_checkpoint_csv_log(metrics_csv)
        init_step_csv_log(step_metrics_csv)
        init_auto_freq_csv_log(auto_freq_csv)

        r0_print(f"Training Start. Safe full-file mode. experiment_name={args.experiment_name}")

        if args.log_disk_usage_each_ckpt:
            log_disk_usage(args.output_dir, prefix="[INIT] ")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=None,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )

    base_blocks = load_dataset_stream(args.train_file, tokenizer, block_size=args.block_size)
    required_steps = args.required_steps

    if len(base_blocks) > 0:
        repeated = []
        while len(repeated) < required_steps:
            repeated.extend(base_blocks)
        train_blocks = repeated[:required_steps]
    else:
        train_blocks = [torch.zeros((args.block_size,), dtype=torch.long) for _ in range(required_steps)]

    if is_env_rank0():
        r0_print("[INIT] before deepspeed.initialize()")

    ds_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    if ds_engine.global_rank == 0:
        r0_print("[INIT] before Checkpointing()")

    checkpointing = Checkpointing(runtime_config=ds_ckpt_cfg, rank=ds_engine.global_rank)

    if ds_engine.global_rank == 0:
        r0_print("[INIT] after Checkpointing()")

    ensure_ckpt_engine_stats_snapshot(checkpointing, ds_ckpt_cfg)

    if ds_engine.global_rank == 0:
        r0_print("[INIT] after ensure_ckpt_engine_stats_snapshot()")

    if args.durable_mode == "auto":
        durable_mode_effective = "before_next_ckpt"
    else:
        durable_mode_effective = args.durable_mode

    can_wait_durable = checkpointing_has_wait_durable(checkpointing)

    if durable_mode_effective != "off" and not can_wait_durable:
        raise RuntimeError(
            "[Durability] wait_durable() is missing.\n"
            "Patch engine.hpp / engine.cpp / py_datastates_llm.cpp, rebuild, then rerun."
        )

    if args.keep_latest_n > 0 and durable_mode_effective == "off":
        raise RuntimeError(
            "[Safety] keep_latest_n > 0 with durable_mode=off is unsafe.\n"
            "Use --durable_mode auto / before_next_ckpt / each_ckpt."
        )

    if args.keep_recent_n_during_run > 0 and durable_mode_effective == "off":
        raise RuntimeError(
            "[Safety] keep_recent_n_during_run > 0 with durable_mode=off is unsafe.\n"
            "Use --durable_mode auto / before_next_ckpt / each_ckpt."
        )

    if args.delete_all_checkpoints_at_end == 1 and not can_wait_durable:
        raise RuntimeError(
            "[Safety] delete_all_checkpoints_at_end=1 requires wait_durable().\n"
            "Patch engine.hpp / engine.cpp / py_datastates_llm.cpp, rebuild, then rerun."
        )

    if ds_engine.global_rank == 0:
        r0_print(
            f"[INIT] keep_latest_n={args.keep_latest_n} | "
            f"keep_recent_n_during_run={args.keep_recent_n_during_run} | "
            f"delete_all_checkpoints_at_end={args.delete_all_checkpoints_at_end} | "
            f"retention_cleanup_mode={args.retention_cleanup_mode} | "
            f"durable_mode={args.durable_mode} -> effective={durable_mode_effective}"
        )

        write_json(run_config_json, {
            "experiment_name": args.experiment_name,
            "args": vars(args),
            "resolved": {
                "durable_mode_effective": durable_mode_effective,
                "can_wait_durable": can_wait_durable,
                "world_size": dist_world_size(),
                "runtime_config": ds_ckpt_cfg,
            },
        })

    start_epoch, start_step = 0, 0
    if args.resume_from:
        tag_dir = os.path.join(ckpt_dir, args.resume_from)
        model_file = os.path.join(tag_dir, "mp_rank_00_model_states.pt")
        if ds_engine.global_rank == 0 and os.path.exists(model_file):
            model_state = checkpointing.load(model_file)
            load_state = maybe_restore_tied_output_head(model_state["model"])
            ds_engine.load_state_dict(load_state)
        dist_barrier()
        if args.resume_from.startswith("step_"):
            start_step = int(args.resume_from.split("_")[1])

    auto_freq = AutoCheckFreq(
        target_overhead=args.auto_target_overhead,
        initial_freq=args.initial_freq,
        profile_steps=args.auto_profile_steps,
    ) if args.enable_auto_freq else None

    if ds_engine.global_rank == 0:
        r0_print("[INIT] entering training loop")

    pending_durable_tag = None
    zero_stats = _normalize_ckpt_stats({}, ds_ckpt_cfg)

    try:
        for epoch in range(start_epoch, args.epochs):
            for step, block in enumerate(train_blocks):
                if epoch == start_epoch and step < start_step:
                    continue

                step_index = step + 1
                step_wall_begin = time.time()

                checkpoint_requested_initial = False
                checkpoint_requested_final = False
                checkpoint_executed = False
                checkpoint_skipped_by_backpressure = False
                checkpoint_reason = "none"
                ckpt_tag = ""

                wait_prev_durable_triggered = False
                wait_prev_durable_local_seconds = 0.0
                wait_prev_durable_window_seconds = 0.0
                wait_prev_durable_bp_wait_seconds = 0.0
                pre_ckpt_backpressure_wait_seconds = 0.0

                total_ckpt_seconds = 0.0
                enqueue_seconds = 0.0
                flush_barrier_seconds = 0.0
                real_flush_duration = 0.0
                durable_local_seconds = 0.0
                durable_window_seconds = 0.0

                m_elements, m_g2c, m_c2d = 0, 0.0, 0.0
                o_elements, o_g2c, o_c2d = 0, 0.0, 0.0

                cleanup_triggered_this_step = False
                cleanup_deleted_count_this_step = 0
                cleanup_seconds_this_step = 0.0
                cleanup_reason_this_step = ""

                stats_t1 = dict(zero_stats)
                stats_t2 = dict(zero_stats)
                process_rss_bytes_t1, process_vmlck_bytes_t1 = 0, 0
                process_rss_bytes_t2, process_vmlck_bytes_t2 = 0, 0

                pending_durable_tag_before = pending_durable_tag or ""

                # ------------------------------------------------------------
                # Forward / Backward / Optimizer
                # ------------------------------------------------------------
                t_iter_start = time.time()
                t_bw_start = time.time()

                inputs = block.unsqueeze(0).to(ds_engine.device)
                loss = ds_engine(inputs, labels=inputs.clone()).loss
                ds_engine.backward(loss)

                t_bw_end = time.time()

                t_step_start = time.time()
                ds_engine.step()
                t_step_end = time.time()

                iter_seconds = time.time() - t_iter_start
                backward_seconds = t_bw_end - t_bw_start
                optimizer_step_seconds = t_step_end - t_step_start

                # ------------------------------------------------------------
                # Auto profile logging
                # ------------------------------------------------------------
                if auto_freq is not None:
                    auto_events = auto_freq.update_iter_time(iter_seconds, step=step_index)
                    if ds_engine.global_rank == 0:
                        for evt in auto_events:
                            append_csv_row(auto_freq_csv, [
                                datetime.now().isoformat(),
                                args.experiment_name,
                                evt["event_type"],
                                evt["step"],
                                int(evt["is_profiling"]),
                                evt["profiled_steps"],
                                evt["profile_steps"],
                                evt["avg_iter_time_seconds"],
                                evt["target_overhead"],
                                evt["ckpt_duration_seconds"],
                                evt["actual_interval"],
                                evt["overhead_pct"],
                                evt["required_interval"],
                                evt["ckpt_equiv_iters"],
                                evt["old_freq"],
                                evt["new_freq"],
                                evt["change_type"],
                                evt["reason"],
                            ])

                # ------------------------------------------------------------
                # Checkpoint decision
                # ------------------------------------------------------------
                should_save = False
                checkpoint_reason_parts = []

                if auto_freq is not None:
                    auto_should_ckpt = False
                    if ds_engine.global_rank == 0 and auto_freq.should_checkpoint(step_index):
                        auto_should_ckpt = True
                    if dist_is_ready():
                        flag = torch.tensor(
                            [1 if auto_should_ckpt else 0],
                            dtype=torch.int32,
                            device=ds_engine.device,
                        )
                        torch.distributed.broadcast(flag, src=0)
                        auto_should_ckpt = bool(flag.item())
                    should_save = auto_should_ckpt
                    if auto_should_ckpt:
                        checkpoint_reason_parts.append("auto")
                else:
                    interval_should_ckpt = (step_index % args.checkpoint_interval == 0)
                    should_save = interval_should_ckpt
                    if interval_should_ckpt:
                        checkpoint_reason_parts.append("interval")

                is_final_step = (step_index == required_steps)
                if is_final_step:
                    should_save = True
                    checkpoint_reason_parts.append("final")

                checkpoint_requested_initial = bool(should_save)
                checkpoint_reason = "|".join(checkpoint_reason_parts) if checkpoint_reason_parts else "none"
                next_tag = f"step_{step_index}"

                # ------------------------------------------------------------
                # single-inflight durability gate
                # ------------------------------------------------------------
                if should_save and durable_mode_effective == "before_next_ckpt" and pending_durable_tag is not None:
                    wait_prev_durable_triggered = True

                    if ds_engine.global_rank == 0:
                        r0_print(
                            f"[CKPT-Inflight] waiting previous checkpoint durable before new checkpoint | "
                            f"previous={pending_durable_tag} next={next_tag}"
                        )

                    t_prev_durable_begin = time.time()
                    prev_local = checkpointing_wait_durable(checkpointing)
                    dist_barrier()
                    wait_prev_durable_window_seconds = time.time() - t_prev_durable_begin
                    wait_prev_durable_local_seconds = 0.0 if prev_local is None else float(prev_local)

                    timeout_s = None if args.retention_final_wait_timeout_s <= 0 else args.retention_final_wait_timeout_s
                    wait_prev_durable_bp_wait_seconds = wait_for_backpressure_clear(
                        ds_engine=ds_engine,
                        checkpointing=checkpointing,
                        poll_s=max(0.01, args.backpressure_poll_s),
                        timeout_s=timeout_s,
                        label=f"before_new_ckpt_after_{pending_durable_tag}",
                    )
                    dist_barrier()

                    if ds_engine.global_rank == 0:
                        r0_print(
                            f"[CKPT-Inflight] previous durable done | "
                            f"previous={pending_durable_tag} "
                            f"local_wait_durable={wait_prev_durable_local_seconds:.3f}s "
                            f"window={wait_prev_durable_window_seconds:.3f}s "
                            f"bp_wait={wait_prev_durable_bp_wait_seconds:.3f}s"
                        )

                    # 안전 지점이므로 여기서 최근 N개만 유지
                    if args.keep_recent_n_during_run > 0:
                        if ds_engine.global_rank == 0:
                            cleanup_info = cleanup_old_steps_sync(ckpt_dir, args.keep_recent_n_during_run)
                            cleanup_triggered_this_step = True
                            cleanup_deleted_count_this_step += int(cleanup_info["deleted_count"])
                            cleanup_seconds_this_step += float(cleanup_info["duration_seconds"])
                            cleanup_reason_this_step = _append_reason(cleanup_reason_this_step, "before_new_ckpt")
                            r0_print(
                                f"[Retention-Run] prune after previous durable | "
                                f"keep_n={args.keep_recent_n_during_run} "
                                f"deleted={cleanup_info['deleted_count']} "
                                f"duration={cleanup_info['duration_seconds']:.3f}s"
                            )
                        dist_barrier()

                    pending_durable_tag = None

                # ------------------------------------------------------------
                # Optional checkpoint backpressure guard
                # ------------------------------------------------------------
                if should_save and args.enable_ckpt_backpressure:
                    active, local_bp = global_backpressure_active(ds_engine, checkpointing)
                    if active:
                        if args.backpressure_mode == "skip" and not is_final_step:
                            if ds_engine.global_rank == 0:
                                r0_print(
                                    f"[Backpressure] step={step_index} scheduled ckpt skipped | "
                                    f"local_pending_flush={local_bp['pending_flush_bytes']} "
                                    f"({_human_bytes(local_bp['pending_flush_bytes'])}) | "
                                    f"local_q={local_bp['flush_queue_depth']}"
                                )
                            should_save = False
                            checkpoint_skipped_by_backpressure = True
                        else:
                            timeout_s = None if args.backpressure_timeout_s <= 0 else args.backpressure_timeout_s
                            pre_ckpt_backpressure_wait_seconds = wait_for_backpressure_clear(
                                ds_engine=ds_engine,
                                checkpointing=checkpointing,
                                poll_s=args.backpressure_poll_s,
                                timeout_s=timeout_s,
                                label=f"before_ckpt_step_{step_index}",
                            )
                            if ds_engine.global_rank == 0:
                                r0_print(
                                    f"[Backpressure] step={step_index} waited="
                                    f"{pre_ckpt_backpressure_wait_seconds:.3f}s before checkpoint"
                                )

                checkpoint_requested_final = bool(should_save)

                # ------------------------------------------------------------
                # Checkpoint execution
                # ------------------------------------------------------------
                if should_save:
                    checkpoint_executed = True
                    ckpt_tag = next_tag
                    tag_dir = os.path.join(ckpt_dir, ckpt_tag)

                    if ds_engine.global_rank == 0:
                        os.makedirs(tag_dir, exist_ok=True)
                        r0_print(f"[STEP-CKPT] epoch={epoch+1} step={step_index}/{required_steps} tag={ckpt_tag} begin")
                        if args.log_disk_usage_each_ckpt:
                            log_disk_usage(args.output_dir, prefix=f"[Before-CKPT {ckpt_tag}] ")

                    dist_barrier()

                    t_save_begin = time.time()

                    # 1) model save rank0
                    if ds_engine.global_rank == 0:
                        raw_state = ds_engine.state_dict()
                        raw_state, _ = maybe_strip_tied_output_head(raw_state)

                        t0 = time.time()
                        m_cpu, _, m_elements = move_state_to_cpu(raw_state, "model")
                        m_g2c = time.time() - t0

                        t0 = time.time()
                        checkpointing.save({"model": m_cpu}, os.path.join(tag_dir, "mp_rank_00_model_states.pt"))
                        m_c2d = time.time() - t0

                    # 2) optimizer save all ranks
                    raw_opt = optimizer.state_dict()

                    t0 = time.time()
                    o_cpu, _, o_elements = move_state_to_cpu(raw_opt, f"optim_rank{ds_engine.global_rank}")
                    o_g2c = time.time() - t0

                    t0 = time.time()
                    checkpointing.save(
                        {"optimizer": o_cpu},
                        os.path.join(tag_dir, f"zero_pp_rank_{ds_engine.global_rank}_mp_rank_00_optim_states.pt"),
                    )
                    o_c2d = time.time() - t0

                    t_after_enqueue = time.time()

                    stats_t1 = checkpointing.stats_snapshot()
                    process_rss_bytes_t1, process_vmlck_bytes_t1 = _read_proc_status_memory_bytes()

                    stats_t2 = checkpointing.stats_snapshot()
                    process_rss_bytes_t2, process_vmlck_bytes_t2 = _read_proc_status_memory_bytes()

                    if ds_engine.global_rank == 0:
                        r0_print(
                            f"[CKPT-Stats] epoch={epoch+1} step={step_index}/{required_steps} tag={ckpt_tag} "
                            f"T1(cap={stats_t1['host_cache_capacity_bytes']}, "
                            f"used={stats_t1['host_cache_used_bytes']}, "
                            f"pending={stats_t1['pending_flush_bytes']}, "
                            f"q={stats_t1['flush_queue_depth']}, "
                            f"rss={process_rss_bytes_t1}, vmlck={process_vmlck_bytes_t1}) | "
                            f"T2(cap={stats_t2['host_cache_capacity_bytes']}, "
                            f"used={stats_t2['host_cache_used_bytes']}, "
                            f"pending={stats_t2['pending_flush_bytes']}, "
                            f"q={stats_t2['flush_queue_depth']}, "
                            f"rss={process_rss_bytes_t2}, vmlck={process_vmlck_bytes_t2})"
                        )
                        if args.log_disk_usage_each_ckpt:
                            log_disk_usage(args.output_dir, prefix=f"[After-Enqueue {ckpt_tag}] ")

                    # 3) hot-path wait
                    real_flush_duration = checkpointing_wait_background(checkpointing)
                    if real_flush_duration is None:
                        real_flush_duration = 0.0

                    dist_barrier()
                    t_after_wait = time.time()

                    # 4) durability policy
                    if durable_mode_effective == "each_ckpt":
                        t_durable_window_begin = time.time()
                        durable_local = checkpointing_wait_durable(checkpointing)
                        durable_local_seconds = 0.0 if durable_local is None else float(durable_local)
                        dist_barrier()
                        durable_window_seconds = time.time() - t_durable_window_begin

                        timeout_s = None if args.retention_final_wait_timeout_s <= 0 else args.retention_final_wait_timeout_s
                        wait_for_backpressure_clear(
                            ds_engine=ds_engine,
                            checkpointing=checkpointing,
                            poll_s=max(0.01, args.backpressure_poll_s),
                            timeout_s=timeout_s,
                            label=f"after_ckpt_{ckpt_tag}",
                        )
                        dist_barrier()

                        # 안전 지점이므로 여기서 최근 N개만 유지
                        if args.keep_recent_n_during_run > 0:
                            if ds_engine.global_rank == 0:
                                cleanup_info = cleanup_old_steps_sync(ckpt_dir, args.keep_recent_n_during_run)
                                cleanup_triggered_this_step = True
                                cleanup_deleted_count_this_step += int(cleanup_info["deleted_count"])
                                cleanup_seconds_this_step += float(cleanup_info["duration_seconds"])
                                cleanup_reason_this_step = _append_reason(cleanup_reason_this_step, "after_ckpt")
                                r0_print(
                                    f"[Retention-Run] prune after checkpoint durable | "
                                    f"keep_n={args.keep_recent_n_during_run} "
                                    f"deleted={cleanup_info['deleted_count']} "
                                    f"duration={cleanup_info['duration_seconds']:.3f}s"
                                )
                            dist_barrier()

                        pending_durable_tag = None

                        if ds_engine.global_rank == 0:
                            r0_print(
                                f"[CKPT-Durable] epoch={epoch+1} step={step_index}/{required_steps} tag={ckpt_tag} "
                                f"local_wait_durable={durable_local_seconds:.3f}s "
                                f"window={durable_window_seconds:.3f}s"
                            )
                    else:
                        pending_durable_tag = ckpt_tag

                    t_save_end = time.time()

                    total_ckpt_seconds = t_save_end - t_save_begin
                    enqueue_seconds = t_after_enqueue - t_save_begin
                    flush_barrier_seconds = t_after_wait - t_after_enqueue

                    # auto freq cost logging
                    if auto_freq is not None and ds_engine.global_rank == 0:
                        evt = auto_freq.update_ckpt_cost(total_ckpt_seconds, step_index)
                        append_csv_row(auto_freq_csv, [
                            datetime.now().isoformat(),
                            args.experiment_name,
                            evt["event_type"],
                            evt["step"],
                            int(evt["is_profiling"]),
                            evt["profiled_steps"],
                            evt["profile_steps"],
                            evt["avg_iter_time_seconds"],
                            evt["target_overhead"],
                            evt["ckpt_duration_seconds"],
                            evt["actual_interval"],
                            evt["overhead_pct"],
                            evt["required_interval"],
                            evt["ckpt_equiv_iters"],
                            evt["old_freq"],
                            evt["new_freq"],
                            evt["change_type"],
                            evt["reason"],
                        ])

                    # checkpoint-only CSV
                    if ds_engine.global_rank == 0:
                        append_csv_row(metrics_csv, [
                            datetime.now().isoformat(),
                            args.experiment_name,
                            epoch + 1,
                            step_index,
                            loss.item(),
                            ds_engine.global_rank,
                            total_ckpt_seconds,
                            enqueue_seconds,
                            flush_barrier_seconds,
                            durable_local_seconds,
                            durable_window_seconds,
                            cleanup_seconds_this_step,
                            cleanup_deleted_count_this_step,
                            m_g2c,
                            m_c2d,
                            o_g2c,
                            o_c2d,
                            m_elements,
                            o_elements,
                            m_elements * 2,
                            o_elements * 4,
                            (m_elements * 2 / total_ckpt_seconds / 1e9 if total_ckpt_seconds > 0 else 0.0),
                            (o_elements * 4 / total_ckpt_seconds / 1e9 if total_ckpt_seconds > 0 else 0.0),
                            backward_seconds,
                            optimizer_step_seconds,
                            stats_t1["host_cache_capacity_bytes"],
                            stats_t1["host_cache_used_bytes"],
                            stats_t2["host_cache_used_bytes"],
                            stats_t1["pending_flush_bytes"],
                            stats_t2["pending_flush_bytes"],
                            stats_t1["flush_queue_depth"],
                            stats_t2["flush_queue_depth"],
                            process_rss_bytes_t1,
                            process_rss_bytes_t2,
                            process_vmlck_bytes_t1,
                            process_vmlck_bytes_t2,
                        ])

                        hotwait_str = f"{float(real_flush_duration):.2f}s" if float(real_flush_duration) > 0 else "N/A"
                        if durable_mode_effective == "each_ckpt":
                            durable_label = f"each_ckpt({durable_window_seconds:.2f}s)"
                        elif durable_mode_effective == "before_next_ckpt":
                            durable_label = "before_next_ckpt(pending)"
                        else:
                            durable_label = "off"

                        r0_print(
                            f"[CKPT-Done] epoch={epoch+1} step={step_index}/{required_steps} tag={ckpt_tag} "
                            f"Total={total_ckpt_seconds:.2f}s "
                            f"(HotWait={hotwait_str}, Durable={durable_label}, "
                            f"CleanupSeconds={cleanup_seconds_this_step:.3f}, CleanupDeleted={cleanup_deleted_count_this_step}, "
                            f"CleanupMode={args.retention_cleanup_mode})"
                        )

                # ------------------------------------------------------------
                # Per-step end logging
                # ------------------------------------------------------------
                step_total_wall_seconds = time.time() - step_wall_begin
                post_compute_overhead_seconds = step_total_wall_seconds - iter_seconds
                pending_durable_tag_after = pending_durable_tag or ""

                auto_snapshot = auto_freq.snapshot() if auto_freq is not None else {
                    "is_profiling": False,
                    "profiled_steps": 0,
                    "profile_steps": 0,
                    "avg_iter_time_seconds": 0.0,
                    "current_freq": 0,
                    "last_ckpt_step": 0,
                    "target_overhead": 0.0,
                    "last_cost_eval": {
                        "step": None,
                        "ckpt_duration_seconds": 0.0,
                        "actual_interval": 0,
                        "overhead_pct": 0.0,
                        "required_interval": 0,
                        "ckpt_equiv_iters": 0.0,
                        "old_freq": 0,
                        "new_freq": 0,
                        "change_type": "none",
                    },
                }

                if ds_engine.global_rank == 0:
                    last_eval = auto_snapshot["last_cost_eval"]

                    append_csv_row(step_metrics_csv, [
                        datetime.now().isoformat(),
                        args.experiment_name,
                        epoch + 1,
                        step_index,
                        ds_engine.global_rank,
                        loss.item(),
                        iter_seconds,
                        step_total_wall_seconds,
                        post_compute_overhead_seconds,
                        backward_seconds,
                        optimizer_step_seconds,
                        int(checkpoint_requested_initial),
                        checkpoint_reason,
                        int(checkpoint_requested_final),
                        int(checkpoint_executed),
                        int(checkpoint_skipped_by_backpressure),
                        ckpt_tag,
                        durable_mode_effective,
                        pending_durable_tag_before,
                        pending_durable_tag_after,
                        int(wait_prev_durable_triggered),
                        wait_prev_durable_local_seconds,
                        wait_prev_durable_window_seconds,
                        wait_prev_durable_bp_wait_seconds,
                        pre_ckpt_backpressure_wait_seconds,
                        total_ckpt_seconds,
                        enqueue_seconds,
                        flush_barrier_seconds,
                        float(real_flush_duration),
                        durable_local_seconds,
                        durable_window_seconds,
                        int(cleanup_triggered_this_step),
                        cleanup_deleted_count_this_step,
                        cleanup_seconds_this_step,
                        cleanup_reason_this_step,
                        stats_t1["host_cache_capacity_bytes"],
                        stats_t1["host_cache_used_bytes"],
                        stats_t2["host_cache_used_bytes"],
                        stats_t1["pending_flush_bytes"],
                        stats_t2["pending_flush_bytes"],
                        stats_t1["flush_queue_depth"],
                        stats_t2["flush_queue_depth"],
                        process_rss_bytes_t1,
                        process_rss_bytes_t2,
                        process_vmlck_bytes_t1,
                        process_vmlck_bytes_t2,
                        int(auto_freq is not None),
                        int(auto_snapshot["is_profiling"]),
                        auto_snapshot["profiled_steps"],
                        auto_snapshot["profile_steps"],
                        auto_snapshot["target_overhead"],
                        auto_snapshot["avg_iter_time_seconds"],
                        auto_snapshot["current_freq"],
                        auto_snapshot["last_ckpt_step"],
                        last_eval["step"],
                        last_eval["ckpt_duration_seconds"],
                        last_eval["actual_interval"],
                        last_eval["overhead_pct"],
                        last_eval["required_interval"],
                        last_eval["ckpt_equiv_iters"],
                        last_eval["old_freq"],
                        last_eval["new_freq"],
                        last_eval["change_type"],
                    ])

                    auto_part = ""
                    if auto_freq is not None:
                        auto_part = (
                            f" auto_freq={auto_snapshot['current_freq']}"
                            f" auto_profile={int(auto_snapshot['is_profiling'])}"
                        )

                    r0_print(
                        f"[STEP-END] epoch={epoch+1} step={step_index}/{required_steps} "
                        f"loss={loss.item():.6f} "
                        f"compute={iter_seconds:.3f}s "
                        f"wall={step_total_wall_seconds:.3f}s "
                        f"over={post_compute_overhead_seconds:.3f}s "
                        f"bw={backward_seconds:.3f}s "
                        f"opt={optimizer_step_seconds:.3f}s "
                        f"ckpt_req={int(checkpoint_requested_initial)} "
                        f"ckpt_exec={int(checkpoint_executed)} "
                        f"cleanup={cleanup_seconds_this_step:.3f}s "
                        f"cleanup_deleted={cleanup_deleted_count_this_step}"
                        f"{auto_part}"
                    )

        # ---------------------------------------------------------------------
        # Final drain / cleanup
        # ---------------------------------------------------------------------
        final_durable_local_seconds = 0.0
        final_durable_window_seconds = 0.0

        if pending_durable_tag is not None and durable_mode_effective in ("before_next_ckpt", "off"):
            if can_wait_durable:
                if ds_engine.global_rank == 0:
                    r0_print(f"[FINALIZE] draining last pending checkpoint durable | tag={pending_durable_tag}")

                t_final_durable_begin = time.time()
                durable_local = checkpointing_wait_durable(checkpointing)
                final_durable_local_seconds = 0.0 if durable_local is None else float(durable_local)
                dist_barrier()
                final_durable_window_seconds = time.time() - t_final_durable_begin

                timeout_s = None if args.retention_final_wait_timeout_s <= 0 else args.retention_final_wait_timeout_s
                bp_wait = wait_for_backpressure_clear(
                    ds_engine=ds_engine,
                    checkpointing=checkpointing,
                    poll_s=max(0.01, args.backpressure_poll_s),
                    timeout_s=timeout_s,
                    label="final_sync_cleanup",
                )
                dist_barrier()

                if ds_engine.global_rank == 0:
                    r0_print(
                        f"[FINALIZE] last durable drain done | "
                        f"tag={pending_durable_tag} "
                        f"local_wait_durable={final_durable_local_seconds:.3f}s "
                        f"window={final_durable_window_seconds:.3f}s "
                        f"bp_wait={bp_wait:.3f}s"
                    )
            else:
                if ds_engine.global_rank == 0:
                    r0_print(
                        f"[FINALIZE-WARN] pending_durable_tag={pending_durable_tag} but wait_durable is unavailable; skip"
                    )

            pending_durable_tag = None

        # 종료 후 checkpoint 정리
        if args.delete_all_checkpoints_at_end == 1:
            # 마지막으로 한 번 더 backlog 없는지 확인
            timeout_s = None if args.retention_final_wait_timeout_s <= 0 else args.retention_final_wait_timeout_s
            bp_wait = wait_for_backpressure_clear(
                ds_engine=ds_engine,
                checkpointing=checkpointing,
                poll_s=max(0.01, args.backpressure_poll_s),
                timeout_s=timeout_s,
                label="final_delete_all_checkpoints",
            )
            dist_barrier()

            if ds_engine.global_rank == 0:
                r0_print(f"[FINALIZE] delete_all_checkpoints_at_end=1 | bp_wait={bp_wait:.3f}s")
                delete_info = delete_all_steps_sync(ckpt_dir)
                r0_print(
                    f"[FINALIZE] deleted all step checkpoints | "
                    f"deleted={delete_info['deleted_count']} "
                    f"duration={delete_info['duration_seconds']:.3f}s"
                )
            dist_barrier()

        elif args.keep_latest_n > 0 and args.retention_cleanup_mode != "none":
            if ds_engine.global_rank == 0:
                r0_print("[FINALIZE] keep_latest_n cleanup begin ...")
                cleanup_info = cleanup_old_steps_sync(ckpt_dir, keep_n=args.keep_latest_n)
                r0_print(
                    f"[FINALIZE] keep_latest_n cleanup done | "
                    f"keep_n={args.keep_latest_n} "
                    f"deleted={cleanup_info['deleted_count']} "
                    f"duration={cleanup_info['duration_seconds']:.3f}s"
                )
            dist_barrier()

        dist_barrier()
        r0_print("Training finished.")

    finally:
        try:
            checkpointing_close_safely(checkpointing)
        except Exception as e:
            print(f"[Checkpointing-Close-Error] {e}", flush=True)


if __name__ == "__main__":
    main()