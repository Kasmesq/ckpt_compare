import torch
from concurrent.futures import ThreadPoolExecutor
import time
from collections import OrderedDict, deque
import sys
import os
from typing import Union
import pickle
import json
import ctypes
import numpy as np

from datastates.ckpt import CkptEngine
# helper 모듈 import 경로가 환경에 따라 다를 수 있으므로 try-except 없이 원본 유지하거나
# 필요하다면 아래처럼 import 경로를 확인해주세요. (원본 코드의 import 유지)
from .helper import parse_config, get_checkpoint_version, HOST_CACHE_SIZE, CKPT_PARSER_THREADS
from datastates.utils import get_logger

SIZE_UINT64 = ctypes.sizeof(ctypes.c_uint64)
KEY_SEPARATOR = "|"

class Checkpointing:
    def __init__(self, runtime_config={}, rank=0) -> None:
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("[DataStates.llm] CUDA is not available. Make sure CUDA drivers are installed and GPU is accessible.")

            self.rank = int(rank)
            datastates_config = parse_config(runtime_config)

            host_cache_size = int(datastates_config[HOST_CACHE_SIZE] * (1 << 30))
            cuda_device = int(torch.cuda.current_device())
            concurrent_parser_threads = int(datastates_config[CKPT_PARSER_THREADS])

            self.ckpt_engine = CkptEngine(host_cache_size, cuda_device, self.rank)
            self.executor = ThreadPoolExecutor(max_workers=concurrent_parser_threads)
            self.logger = get_logger(__name__)
            self.last_ckpt_version = -1

            self.debug_timing = True

            # ✅ NEW: track inflight futures
            self._pending_futures = deque()

            # ✅ NEW: limit inflight to avoid host-cache saturation (key for your 600/800 regression)
            self.max_inflight_ckpts = int(runtime_config.get("max_inflight_ckpts", 2))
            
            # ★ [추가] Flush 시작 시간 기록용 변수
            self.last_flush_start_time = 0.0

        except Exception as exc:
            print(f"[DataStates.llm][ERROR] Got exception during DataStates init {exc}")
            sys.exit(-1)

    def _maybe_backpressure(self):
        # if too many inflight, wait the oldest one to finish parse+enqueue
        while len(self._pending_futures) > self.max_inflight_ckpts:
            fut = self._pending_futures.popleft()
            if self.debug_timing:
                self.logger.info("[DataStates.llm][CKPT-TIMING] backpressure: wait oldest inflight future")
            fut.result()  # raise if error
            # optional: free internal buffers earlier (prevents async_save enqueue from blocking later)
            self.ckpt_engine.wait()

    def save_background(self, state_dict: Union[dict, OrderedDict], path: str):
        try:
            if self.debug_timing:
                self.logger.info(f"[DataStates.llm][CKPT-TIMING] save_background START path={path}")

            t_total_begin = time.time()

            version = get_checkpoint_version(path, self.last_ckpt_version)
            header = {}
            async_copies = {}
            _start_tensor_offset = 0
            _end_tensor_offset = 0

            def _parse_state(key, data):
                nonlocal _start_tensor_offset, _end_tensor_offset
                if torch.is_tensor(data):
                    tensor_size = data.numel() * data.element_size()
                    _end_tensor_offset += tensor_size
                    header[key] = {
                        "dtype": str(data.dtype),
                        "shape": tuple(data.shape),
                        "data_offsets": [_start_tensor_offset, _end_tensor_offset],
                    }
                    data = data.contiguous()
                    async_copies[key] = {"tensor": data, "file_offset": _start_tensor_offset}
                    _start_tensor_offset = _end_tensor_offset
                    return f"TENSOR{KEY_SEPARATOR}{key}"

                elif isinstance(data, list):
                    snapshot = [None] * len(data)
                    for idx, ele in enumerate(data):
                        new_key = f"{key}{KEY_SEPARATOR}{idx}" if len(key) else f"{idx}"
                        snapshot[idx] = _parse_state(new_key, ele)
                    return snapshot

                elif isinstance(data, (dict, OrderedDict)):
                    snapshot = {}
                    for k, v in data.items():
                        new_key = f"{key}{KEY_SEPARATOR}{k}" if len(key) else f"{k}"
                        snapshot[k] = _parse_state(new_key, v)
                    return snapshot

                else:
                    return data

            # 1) parse+pickle
            t_parse_begin = time.time()
            lean_state_dict = _parse_state("", state_dict)
            lean_state_bytes = pickle.dumps(lean_state_dict, protocol=pickle.HIGHEST_PROTOCOL)
            t_parse_end = time.time()

            _end_tensor_offset += len(lean_state_bytes)

            header.update({"datastates_metadata": {"data_offsets": [_start_tensor_offset, _end_tensor_offset]}})
            header_bytes = json.dumps(header).encode("utf-8")
            header_size_bytes = len(header_bytes).to_bytes(SIZE_UINT64, "little")
            metadata_size = len(header_size_bytes) + len(header_bytes)

            # 2) write header + lean_state first (stable file layout)
            t_file_begin = time.time()
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as f:
                f.seek(0)
                f.write(header_size_bytes)
                f.write(header_bytes)
                f.seek(metadata_size + _start_tensor_offset)
                f.write(lean_state_bytes)
            t_file_end = time.time()

            # 3) enqueue tensors
            async_ckpt_list = []
            for _, v in async_copies.items():
                v["file_offset"] += metadata_size
                async_ckpt_list.append((version, v["tensor"], v["file_offset"], path))

            t_async_begin = time.time()
            
            # ★ [추가] 디스크 쓰기 시작 시간 갱신
            if async_ckpt_list:
                self.last_flush_start_time = time.time()
                self.ckpt_engine.async_save(async_ckpt_list)
            t_async_end = time.time()

            t_total_end = time.time()

            if self.debug_timing:
                self.logger.info(
                    f"[DataStates.llm][CKPT-TIMING] save_background DONE path={path} | "
                    f"parse+pickle={t_parse_end - t_parse_begin:.3f}s, "
                    f"file_write(header+lean)={t_file_end - t_file_begin:.3f}s, "
                    f"async_save_enqueue={t_async_end - t_async_begin:.3f}s, "
                    f"total={t_total_end - t_total_begin:.3f}s"
                )

            return True

        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] From save_background exception: {exc}")
            raise

    def save(self, state_dict, path: str):
        try:
            if not isinstance(state_dict, (dict, OrderedDict)):
                raise Exception(f"[DataStates.llm] state_dict must be dict/OrderedDict. Got {type(state_dict)} for {path}.")

            if self.debug_timing:
                self.logger.info(f"[DataStates.llm][CKPT-TIMING] submit save_background path={path}")

            fut = self.executor.submit(self.save_background, state_dict, path)
            self._pending_futures.append(fut)

            # ✅ 핵심: 너무 많이 쌓이면 미리 정리해서 enqueue 폭발 막기
            self._maybe_backpressure()

            return True

        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] Could not save {path}, exception: {exc}")
            sys.exit(-1)

    def wait(self):
        try:
            if self.debug_timing:
                self.logger.info("[DataStates.llm][CKPT-TIMING] wait() START")

            # 1) join python futures
            join_cnt = 0
            while self._pending_futures:
                fut = self._pending_futures.popleft()
                fut.result()
                join_cnt += 1

            # 2) engine flush
            engine_elapsed = self.ckpt_engine.wait()

            # 3) compute flush_duration safely
            now = time.time()
            start = float(getattr(self, "last_flush_start_time", 0.0) or 0.0)

            if start <= 0.0:
                flush_duration = 0.0
            else:
                flush_duration = now - start
                # sanity guard
                if flush_duration < 0.0 or flush_duration > 3600.0:  # 1 hour cap
                    flush_duration = 0.0

            # reset so later waits don't reuse stale start time
            self.last_flush_start_time = 0.0

            if self.debug_timing:
                self.logger.info(
                    f"[DataStates.llm][CKPT-TIMING] wait() DONE | "
                    f"joined={join_cnt} | "
                    f"engine_elapsed={0.0 if engine_elapsed is None else float(engine_elapsed):.4f}s | "
                    f"Duration={flush_duration:.4f}s"
                )

            return flush_duration

        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] From wait, generated exception: {exc}")
            sys.exit(-1)
            
    def commit(self, tag):
        self.wait_durable()
        self.logger.info(f"[DataStates.llm] Checkpoint {tag} is durable now!")
        self.last_ckpt_version += 1
        return True


    def _resolve_wait_durable_callable(self):
        """
        CkptEngine wrapper에 직접 wait_durable()가 있으면 그걸 쓰고,
        없으면 내부 pybind handle에서 찾아서 쓴다.
        wrapper 구현이 환경마다 조금 달라도 최대한 버티게 만든다.
        """
        # 1) wrapper 자체에 있으면 바로 사용
        fn = getattr(self.ckpt_engine, "wait_durable", None)
        if callable(fn):
            return fn

        # 2) 내부 handle 후보들 탐색
        for attr_name in (
            "handle",
            "datastates_handle",
            "_handle",
            "engine",
            "_engine",
            "ckpt_engine",
            "_ckpt_engine",
        ):
            inner = getattr(self.ckpt_engine, attr_name, None)
            if inner is None:
                continue

            fn = getattr(inner, "wait_durable", None)
            if callable(fn):
                return fn

        return None

    def has_wait_durable(self):
        return self._resolve_wait_durable_callable() is not None

    def wait_durable(self):
        try:
            if self.debug_timing:
                self.logger.info("[DataStates.llm][CKPT-TIMING] wait_durable() START")

            # 1) 먼저 Python futures 전부 join
            join_cnt = 0
            while self._pending_futures:
                fut = self._pending_futures.popleft()
                fut.result()
                join_cnt += 1

            # 2) 엔진의 durable wait 호출
            wait_durable_fn = self._resolve_wait_durable_callable()
            if wait_durable_fn is None:
                raise RuntimeError(
                    "[DataStates.llm] wait_durable() is not available on ckpt engine"
                )

            t0 = time.time()
            result = wait_durable_fn()
            elapsed = time.time() - t0

            if self.debug_timing:
                self.logger.info(
                    f"[DataStates.llm][CKPT-TIMING] wait_durable() DONE | "
                    f"joined={join_cnt} | Duration={elapsed:.4f}s"
                )

            # C++가 void면 elapsed를 넘겨주고,
            # 나중에 C++가 숫자 리턴하도록 바뀌면 그 값을 우선 사용
            return result if result is not None else elapsed

        except Exception as exc:
            self.logger.error(
                f"[DataStates.llm][ERROR] From wait_durable, generated exception: {exc}"
            )
            sys.exit(-1)
            
    def close(self):
        # idempotent: 두 번 호출돼도 안전해야 함
        if getattr(self, "_closed", False):
            return
        self._closed = True

        # best-effort drain (절대 여기서 죽이면 안 됨)
        try:
            if self.has_wait_durable():
                self.wait_durable()
            else:
                self.wait()
        except Exception:
            pass

        # executor 종료
        try:
            if hasattr(self, "executor") and self.executor is not None:
                self.executor.shutdown(True)
                self.executor = None
        except Exception:
            pass

        # engine 종료
        try:
            if hasattr(self, "ckpt_engine") and self.ckpt_engine is not None:
                if hasattr(self.ckpt_engine, "close"):
                    self.ckpt_engine.close()
                self.ckpt_engine = None
        except Exception:
            pass
        
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass