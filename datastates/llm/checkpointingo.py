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
            
#             self.rank           = int(rank)
#             datastates_config   = parse_config(runtime_config)
#             host_cache_size     = int(datastates_config[HOST_CACHE_SIZE]*(1<<30))       # From GB to Bytes
#             cuda_device         = int(torch.cuda.current_device())
#             concurrent_parser_threads = int(datastates_config[CKPT_PARSER_THREADS])

#             self.ckpt_engine = CkptEngine(host_cache_size, cuda_device, self.rank)   
#             self.executor    = ThreadPoolExecutor(max_workers=concurrent_parser_threads)
#             self.logger      = get_logger(__name__)
#             self.last_ckpt_version = -1

#             # ===== 타이밍 디버그 플래그 (원하면 runtime_config로 제어해도 됨) =====
#             self.debug_timing = True

#         except Exception as exc:
#             print(f"[DataStates.llm][ERROR] Got exception during DataStates init {exc}")
#             sys.exit(-1)

#     # ------------------------------------------------------------------
#     # 내부: 실제 저장 로직 (동기) - executor에서 호출
#     # ------------------------------------------------------------------
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
#                 try:
#                     if torch.is_tensor(data): # and data.device.type == 'cuda':
#                         tensor_size = data.numel()*data.element_size()
#                         _end_tensor_offset += tensor_size
#                         header[key] = {
#                             "dtype": str(data.dtype),
#                             "shape": tuple(data.shape),
#                             "data_offsets": [_start_tensor_offset, _end_tensor_offset],
#                         }
#                         data = data.contiguous()
#                         async_copies[key] = {
#                             "tensor": data,
#                             "file_offset": _start_tensor_offset
#                         }
#                         _start_tensor_offset = _end_tensor_offset
#                         snapshot = f"TENSOR{KEY_SEPARATOR}{key}"
#                     elif isinstance(data, list):
#                         snapshot = [None]*len(data)
#                         for (idx, ele) in enumerate(data):
#                             new_key = f"{key}{KEY_SEPARATOR}{idx}" if len(key) else f"{idx}"
#                             snapshot[idx] = _parse_state(new_key, ele)
#                     elif isinstance(data, (dict, OrderedDict)):
#                         snapshot = {}
#                         for (k, v) in data.items():
#                             new_key = f"{key}{KEY_SEPARATOR}{k}" if len(key) else f"{k}"
#                             snapshot[k] = _parse_state(new_key, v)
#                     else:
#                         snapshot = data
#                     return snapshot
#                 except Exception as exc:
#                     raise Exception(f"[DataStates.llm][ERROR] Cannot parse {key}, exception: {exc}, data is {data}")

#             # --------- 1) 파싱 + lean_state_dict 직렬화 ---------
#             t_parse_begin = time.time()
#             lean_state_dict = _parse_state("", state_dict)
#             lean_state_dict = pickle.dumps(lean_state_dict, protocol=pickle.HIGHEST_PROTOCOL)
#             t_parse_end = time.time()

#             _end_tensor_offset += len(lean_state_dict)
#             header.update({"datastates_metadata": {"data_offsets": [_start_tensor_offset, _end_tensor_offset]}})
#             header = json.dumps(header).encode("utf-8")
#             header_size = len(header).to_bytes(SIZE_UINT64, 'little')   # Force the header size to take 8 bytes
#             metadata_size = len(header_size) + len(header)

#             # --------- 2) GPU 메모리 → host cache async_save (비동기 enqueue) ---------
#             async_ckpt_list = []
#             for _, v in async_copies.items():
#                 v["file_offset"] += metadata_size
#                 async_ckpt_list.append((version, v["tensor"], v["file_offset"], path))

#             t_async_begin = time.time()
#             self.ckpt_engine.async_save(async_ckpt_list)
#             t_async_end = time.time()

#             # --------- 3) header + lean_state를 파일에 동기 쓰기 ---------
#             t_header_begin = time.time()
#             with open(path, 'wb') as f:
#                 f.seek(0)
#                 f.write(header_size)
#                 f.write(header)
#                 # Write the lean state dict towards the end of the file.
#                 f.seek(_start_tensor_offset+metadata_size)
#                 f.write(lean_state_dict)
#             t_header_end = time.time()

#             t_total_end = time.time()

#             if self.debug_timing:
#                 self.logger.info(
#                     f"[DataStates.llm][CKPT-TIMING] save_background DONE path={path} | "
#                     f"parse+pickle={t_parse_end - t_parse_begin:.3f}s, "
#                     f"async_save_enqueue={t_async_end - t_async_begin:.3f}s, "
#                     f"header+lean_write={t_header_end - t_header_begin:.3f}s, "
#                     f"total_save_background={t_total_end - t_total_begin:.3f}s"
#                 )

#             return None
#         except Exception as exc:
#             self.logger.error(f"[DataStates.llm][ERROR] From DataStates save_background, generated exception: {exc}")
#             sys.exit(-1)

#     # ------------------------------------------------------------------
#     # 외부: 비동기 저장 API (트레이닝 스크립트에서 호출)
#     # ------------------------------------------------------------------
#     def save(self, state_dict, path: str):
#         try:
#             if not isinstance(state_dict, (dict, OrderedDict)):
#                 raise Exception(f"[DataStates.llm] state_dict given to checkpoint must be dictionary. Passed {type(state_dict)} instead for {path}.")

#             if self.debug_timing:
#                 self.logger.info(f"[DataStates.llm][CKPT-TIMING] submit save_background to executor path={path}")

#             self.executor.submit(self.save_background, state_dict, path)
#             # self.save_background(state_dict, path)
#             return True
#         except Exception as exc:
#             self.logger.error(f"[DataStates.llm][ERROR] Could not save {path}, exception: {exc}, data: {state_dict}")
#             sys.exit(-1)

#     # ------------------------------------------------------------------
#     # LOAD
#     # ------------------------------------------------------------------
#     def load(self, path: str, map_location=None):
#         try:
#             if self.debug_timing:
#                 self.logger.info(f"[DataStates.llm][CKPT-TIMING] load START path={path}")

#             t_total_begin = time.time()

#             version = get_checkpoint_version(path, self.last_ckpt_version)
#             f = open(path, 'rb')
#             f.seek(0)
#             header_size_bytes = f.read(SIZE_UINT64)
#             header_size = int.from_bytes(header_size_bytes, 'little')
#             metadata_size = header_size + SIZE_UINT64
#             header = json.loads(f.read(header_size))
#             [start_offset, end_offset] = np.add(header["datastates_metadata"]["data_offsets"], metadata_size)
#             del(header["datastates_metadata"])
#             f.seek(start_offset)
#             data = pickle.loads(f.read(end_offset-start_offset))

#             try:
#                 restore_list = []
#                 for k, v in header.items():
#                     split_k = deque(k.split(KEY_SEPARATOR))
#                     dtype = v["dtype"]
#                     if dtype.startswith("torch"):
#                         dtype = dtype.replace('torch.', '')
#                     shape = v["shape"]
#                     [start_offset, end_offset] = np.add(v["data_offsets"], metadata_size)

#                     pre_dest = data
#                     dest = data
#                     while len(split_k):
#                         sub_k = split_k.popleft()
#                         if sub_k.isdigit():
#                             sub_k = int(sub_k) 
#                         pre_dest = dest
#                         dest = dest[sub_k]
#                     if dest != f"TENSOR{KEY_SEPARATOR}{k}":
#                         raise Exception(f"[DataStates.llm] The key in header {k} does not match key at location {dest}")

#                     tensor_restored = torch.zeros(size=tuple(shape), dtype=getattr(torch, dtype))
#                     restore_list.append((version, tensor_restored, start_offset, path))
#                     pre_dest[sub_k] = tensor_restored

#                 t_async_begin = time.time()
#                 self.ckpt_engine.load(restore_list)
#                 t_async_end = time.time()

#             except Exception as exc:
#                 raise Exception(f"[DataStates.llm] Got error with tensor loading {dtype}, {shape}, {exc}")

#             t_total_end = time.time()

#             if self.debug_timing:
#                 self.logger.info(
#                     f"[DataStates.llm][CKPT-TIMING] load DONE path={path} | "
#                     f"async_load={t_async_end - t_async_begin:.3f}s, "
#                     f"total_load={t_total_end - t_total_begin:.3f}s"
#                 )

#             self.logger.info(f"[DataStates.llm] Loaded checkpoint from {path}.")
#             return data
#         except Exception as exc:
#             self.logger.error(f"[DataStates.llm][ERROR] Could not load {path}, exception: {exc}")
#             sys.exit(-1)

#     # ------------------------------------------------------------------
#     # commit (지금은 안 쓰더라도 그대로 둠)
#     # ------------------------------------------------------------------
#     def commit(self, tag):
#         self.wait()
#         self.logger.info(f"[DataStates.llm] Checkpoint {tag} is ready now!")
#         self.last_ckpt_version += 1
#         return True

#     # ------------------------------------------------------------------
#     # wait: 실제 flush + host cache → disk 동기화 끝날 때까지 블록
#     # ------------------------------------------------------------------
#     def wait(self):
#         try:
#             t = time.time()
#             if self.debug_timing:
#                 self.logger.info("[DataStates.llm][CKPT-TIMING] wait() START")

#             self.ckpt_engine.wait()

#             elapsed = time.time() - t
#             self.logger.info(f"[DataStates.llm] Wait time in checkpointing engine {elapsed}")
#             if self.debug_timing:
#                 self.logger.info(f"[DataStates.llm][CKPT-TIMING] wait() DONE | wait_time={elapsed:.3f}s")
#         except Exception as exc:
#             self.logger.error(f"[DataStates.llm][ERROR] From wait, generated exception: {exc}")
#             sys.exit(-1)
#         return 
    
#     def __del__(self):
#         self.executor.shutdown(True)


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

            # ✅ [수정] 좀비 포인터 방지용: 텐서를 강제로 살려두는 큐
            self.inflight_ref_holder = deque()
            self._pending_futures = deque()
            self.max_inflight_ckpts = int(runtime_config.get("max_inflight_ckpts", 2))

        except Exception as exc:
            print(f"[DataStates.llm][ERROR] Got exception during DataStates init {exc}")
            sys.exit(-1)

    def _maybe_backpressure(self):
        # 1. 완료된 작업 확인
        while len(self._pending_futures) > self.max_inflight_ckpts:
            fut = self._pending_futures.popleft()
            fut.result()  # 에러 체크

        # 2. ✅ [수정] 작업 끝난 텐서 놓아주기 (메모리 해제)
        # C++ 복사가 끝날 때까지 안전하게 잡고 있다가, 큐가 차면 오래된 것부터 해제
        while len(self.inflight_ref_holder) > self.max_inflight_ckpts:
            self.inflight_ref_holder.popleft()

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

            # 1) Parse Metadata
            lean_state_dict = _parse_state("", state_dict)
            lean_state_bytes = pickle.dumps(lean_state_dict, protocol=pickle.HIGHEST_PROTOCOL)
            _end_tensor_offset += len(lean_state_bytes)

            header.update({"datastates_metadata": {"data_offsets": [_start_tensor_offset, _end_tensor_offset]}})
            header_bytes = json.dumps(header).encode("utf-8")
            header_size_bytes = len(header_bytes).to_bytes(SIZE_UINT64, "little")
            metadata_size = len(header_size_bytes) + len(header_bytes)

            # 2) Write Header (Sync)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as f:
                f.seek(0)
                f.write(header_size_bytes)
                f.write(header_bytes)
                f.seek(metadata_size + _start_tensor_offset)
                f.write(lean_state_bytes)

            # 3) Enqueue Tensors (Async)
            async_ckpt_list = []
            for _, v in async_copies.items():
                v["file_offset"] += metadata_size
                async_ckpt_list.append((version, v["tensor"], v["file_offset"], path))

            self.ckpt_engine.async_save(async_ckpt_list)

            if self.debug_timing:
                self.logger.info(f"[DataStates.llm][CKPT-TIMING] Enqueued async save for {path}")

            return True

        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] save_background exception: {exc}")
            sys.exit(-1)

    def save(self, state_dict, path: str):
        try:
            if not isinstance(state_dict, (dict, OrderedDict)):
                raise Exception(f"[DataStates.llm] state_dict must be dict. Got {type(state_dict)}.")

            # ✅ [수정] 텐서를 큐에 넣어 수명을 연장시킵니다!
            self.inflight_ref_holder.append(state_dict)
            self._maybe_backpressure()

            if self.debug_timing:
                self.logger.info(f"[DataStates.llm][CKPT-TIMING] submit save_background path={path}")

            fut = self.executor.submit(self.save_background, state_dict, path)
            self._pending_futures.append(fut)
            return True

        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] Could not save {path}: {exc}")
            sys.exit(-1)

    def wait(self):
        try:
            if self.debug_timing:
                self.logger.info("[DataStates.llm][CKPT-TIMING] wait() START")

            while self._pending_futures:
                self._pending_futures.popleft().result()

            self.ckpt_engine.wait()
            
            # ✅ [수정] 모든 작업이 끝났으니 안전하게 비웁니다.
            self.inflight_ref_holder.clear()

            if self.debug_timing:
                self.logger.info(f"[DataStates.llm][CKPT-TIMING] wait() DONE")

        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] wait exception: {exc}")
            sys.exit(-1)

    def load(self, path: str, map_location=None):
        try:
            version = get_checkpoint_version(path, self.last_ckpt_version)
            f = open(path, 'rb')
            f.seek(0)
            header_size = int.from_bytes(f.read(SIZE_UINT64), 'little')
            metadata_size = header_size + SIZE_UINT64
            header = json.loads(f.read(header_size))
            [start_offset, end_offset] = np.add(header["datastates_metadata"]["data_offsets"], metadata_size)
            del(header["datastates_metadata"])
            f.seek(start_offset)
            data = pickle.loads(f.read(end_offset-start_offset))

            restore_list = []
            for k, v in header.items():
                split_k = deque(k.split(KEY_SEPARATOR))
                dtype = v["dtype"].replace('torch.', '')
                shape = v["shape"]
                [s_off, e_off] = np.add(v["data_offsets"], metadata_size)

                pre_dest = data
                dest = data
                while len(split_k):
                    sub_k = split_k.popleft()
                    if sub_k.isdigit(): sub_k = int(sub_k)
                    pre_dest = dest
                    dest = dest[sub_k]
                
                tensor_restored = torch.zeros(size=tuple(shape), dtype=getattr(torch, dtype))
                restore_list.append((version, tensor_restored, s_off, path))
                pre_dest[sub_k] = tensor_restored
            
            self.ckpt_engine.load(restore_list)
            self.logger.info(f"[DataStates.llm] Loaded checkpoint from {path}.")
            return data
        except Exception as exc:
            self.logger.error(f"[DataStates.llm][ERROR] Could not load {path}: {exc}")
            sys.exit(-1)

    def commit(self, tag):
        self.wait()
        self.logger.info(f"[DataStates.llm] Checkpoint {tag} is ready now!")
        self.last_ckpt_version += 1
        return True

    def __del__(self):
        try:
            self.executor.shutdown(True)
        except:
            pass