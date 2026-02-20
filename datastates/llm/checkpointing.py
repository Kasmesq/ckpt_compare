# import torch
# from concurrent.futures import ThreadPoolExecutor
# import time
# from collections import OrderedDict, deque
# import sys
# import os
# from typing import Union
# import pickle
# import json
# import ctypes
# import numpy as np

# from datastates.ckpt import CkptEngine
# from .helper import parse_config, get_checkpoint_version, HOST_CACHE_SIZE, CKPT_PARSER_THREADS
# from datastates.utils import get_logger

# SIZE_UINT64 = ctypes.sizeof(ctypes.c_uint64)
# KEY_SEPARATOR = "|"

# class Checkpointing:
#     def __init__(self, runtime_config={}, rank=0) -> None:
#         try:
#             if not torch.cuda.is_available():
#                 raise RuntimeError("[DataStates.llm] CUDA is not available. Make sure CUDA drivers are installed and GPU is accessible.")

#             self.rank = int(rank)
#             datastates_config = parse_config(runtime_config)

#             host_cache_size = int(datastates_config[HOST_CACHE_SIZE] * (1 << 30))
#             cuda_device = int(torch.cuda.current_device())
#             concurrent_parser_threads = int(datastates_config[CKPT_PARSER_THREADS])

#             self.ckpt_engine = CkptEngine(host_cache_size, cuda_device, self.rank)
#             self.executor = ThreadPoolExecutor(max_workers=concurrent_parser_threads)
#             self.logger = get_logger(__name__)
#             self.last_ckpt_version = -1

#             self.debug_timing = True

#             # ✅ NEW: track inflight futures
#             self._pending_futures = deque()

#             # ✅ NEW: limit inflight to avoid host-cache saturation (key for your 600/800 regression)
#             self.max_inflight_ckpts = int(runtime_config.get("max_inflight_ckpts", 2))

#         except Exception as exc:
#             print(f"[DataStates.llm][ERROR] Got exception during DataStates init {exc}")
#             sys.exit(-1)

#     def _maybe_backpressure(self):
#         # if too many inflight, wait the oldest one to finish parse+enqueue
#         while len(self._pending_futures) > self.max_inflight_ckpts:
#             fut = self._pending_futures.popleft()
#             if self.debug_timing:
#                 self.logger.info("[DataStates.llm][CKPT-TIMING] backpressure: wait oldest inflight future")
#             fut.result()  # raise if error
#             # optional: free internal buffers earlier (prevents async_save enqueue from blocking later)
#             self.ckpt_engine.wait()

#     def save_background(self, state_dict: Union[dict, OrderedDict], path: str):
#         try:
#             if self.debug_timing:
#                 self.logger.info(f"[DataStates.llm][CKPT-TIMING] save_background START path={path}")

#             t_total_begin = time.time()

#             version = get_checkpoint_version(path, self.last_ckpt_version)
#             header = {}
#             async_copies = {}
#             _start_tensor_offset = 0
#             _end_tensor_offset = 0

#             def _parse_state(key, data):
#                 nonlocal _start_tensor_offset, _end_tensor_offset
#                 if torch.is_tensor(data):
#                     tensor_size = data.numel() * data.element_size()
#                     _end_tensor_offset += tensor_size
#                     header[key] = {
#                         "dtype": str(data.dtype),
#                         "shape": tuple(data.shape),
#                         "data_offsets": [_start_tensor_offset, _end_tensor_offset],
#                     }
#                     data = data.contiguous()
#                     async_copies[key] = {"tensor": data, "file_offset": _start_tensor_offset}
#                     _start_tensor_offset = _end_tensor_offset
#                     return f"TENSOR{KEY_SEPARATOR}{key}"

#                 elif isinstance(data, list):
#                     snapshot = [None] * len(data)
#                     for idx, ele in enumerate(data):
#                         new_key = f"{key}{KEY_SEPARATOR}{idx}" if len(key) else f"{idx}"
#                         snapshot[idx] = _parse_state(new_key, ele)
#                     return snapshot

#                 elif isinstance(data, (dict, OrderedDict)):
#                     snapshot = {}
#                     for k, v in data.items():
#                         new_key = f"{key}{KEY_SEPARATOR}{k}" if len(key) else f"{k}"
#                         snapshot[k] = _parse_state(new_key, v)
#                     return snapshot

#                 else:
#                     return data

#             # 1) parse+pickle
#             t_parse_begin = time.time()
#             lean_state_dict = _parse_state("", state_dict)
#             lean_state_bytes = pickle.dumps(lean_state_dict, protocol=pickle.HIGHEST_PROTOCOL)
#             t_parse_end = time.time()

#             _end_tensor_offset += len(lean_state_bytes)

#             header.update({"datastates_metadata": {"data_offsets": [_start_tensor_offset, _end_tensor_offset]}})
#             header_bytes = json.dumps(header).encode("utf-8")
#             header_size_bytes = len(header_bytes).to_bytes(SIZE_UINT64, "little")
#             metadata_size = len(header_size_bytes) + len(header_bytes)

#             # 2) write header + lean_state first (stable file layout)
#             t_file_begin = time.time()
#             os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
#             with open(path, "wb") as f:
#                 f.seek(0)
#                 f.write(header_size_bytes)
#                 f.write(header_bytes)
#                 f.seek(metadata_size + _start_tensor_offset)
#                 f.write(lean_state_bytes)
#             t_file_end = time.time()

#             # 3) enqueue tensors
#             async_ckpt_list = []
#             for _, v in async_copies.items():
#                 v["file_offset"] += metadata_size
#                 async_ckpt_list.append((version, v["tensor"], v["file_offset"], path))

#             t_async_begin = time.time()
#             self.ckpt_engine.async_save(async_ckpt_list)
#             t_async_end = time.time()

#             t_total_end = time.time()

#             if self.debug_timing:
#                 self.logger.info(
#                     f"[DataStates.llm][CKPT-TIMING] save_background DONE path={path} | "
#                     f"parse+pickle={t_parse_end - t_parse_begin:.3f}s, "
#                     f"file_write(header+lean)={t_file_end - t_file_begin:.3f}s, "
#                     f"async_save_enqueue={t_async_end - t_async_begin:.3f}s, "
#                     f"total={t_total_end - t_total_begin:.3f}s"
#                 )

#             return True

#         except Exception as exc:
#             self.logger.error(f"[DataStates.llm][ERROR] From save_background exception: {exc}")
#             raise

#     def save(self, state_dict, path: str):
#         try:
#             if not isinstance(state_dict, (dict, OrderedDict)):
#                 raise Exception(f"[DataStates.llm] state_dict must be dict/OrderedDict. Got {type(state_dict)} for {path}.")

#             if self.debug_timing:
#                 self.logger.info(f"[DataStates.llm][CKPT-TIMING] submit save_background path={path}")

#             fut = self.executor.submit(self.save_background, state_dict, path)
#             self._pending_futures.append(fut)

#             # ✅ 핵심: 너무 많이 쌓이면 미리 정리해서 enqueue 폭발 막기
#             self._maybe_backpressure()

#             return True

#         except Exception as exc:
#             self.logger.error(f"[DataStates.llm][ERROR] Could not save {path}, exception: {exc}")
#             sys.exit(-1)

#     def wait(self):
#         try:
#             if self.debug_timing:
#                 self.logger.info("[DataStates.llm][CKPT-TIMING] wait() START")

#             # 1) join python futures
#             join_cnt = 0
#             while self._pending_futures:
#                 fut = self._pending_futures.popleft()
#                 fut.result()
#                 join_cnt += 1

#             # 2) engine flush
#             self.ckpt_engine.wait()

#             if self.debug_timing:
#                 self.logger.info(f"[DataStates.llm][CKPT-TIMING] wait() DONE | joined={join_cnt}")

#         except Exception as exc:
#             self.logger.error(f"[DataStates.llm][ERROR] From wait, generated exception: {exc}")
#             sys.exit(-1)

#     def commit(self, tag):
#         self.wait()
#         self.logger.info(f"[DataStates.llm] Checkpoint {tag} is ready now!")
#         self.last_ckpt_version += 1
#         return True

#     def __del__(self):
#         try:
#             self.wait()
#         except Exception:
#             pass
#         try:
#             self.executor.shutdown(True)
#         except Exception:
#             pass

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
            self.ckpt_engine.wait()
            
            # ★ [추가] Flush 완료 시간 측정
            flush_duration = time.time() - self.last_flush_start_time
            if flush_duration < 0: flush_duration = 0.0

            if self.debug_timing:
                # [수정] 로그에 Duration 정보 추가
                self.logger.info(f"[DataStates.llm][CKPT-TIMING] wait() DONE | joined={join_cnt} | Duration={flush_duration:.4f}s")
            
            # ★ [추가] 시간값 리턴 (AutoFreq용)
            return flush_duration

        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] From wait, generated exception: {exc}")
            sys.exit(-1)

    def commit(self, tag):
        self.wait()
        self.logger.info(f"[DataStates.llm] Checkpoint {tag} is ready now!")
        self.last_ckpt_version += 1
        return True

    def __del__(self):
        try:
            self.wait()
        except Exception:
            pass
        try:
            self.executor.shutdown(True)
        except Exception:
            pass