# import os
# import time
# import argparse
# import torch
# import torch.multiprocessing as mp
# import torch.distributed as dist
# import deepspeed
# import multiprocessing as py_mp  # ★ 표준 multiprocessing

# print("[DEBUG] >>> decouple_train_bloom3b.py imported")

# from transformers import AutoModelForCausalLM, AutoTokenizer
# from deepspeed.runtime.checkpoint_engine.decoupled_checkpoint_engine import DecoupledCheckpointEngine

# # -----------------------------------------------------------------------------
# # 🔧 PATCH 0: torch.multiprocessing.get_start_methods 없을 때 추가
# #   - DeepSpeed가 mp.get_start_methods(allow_None=False) 를 호출해서,
# #     시그니처가 (allow_None=...) 를 받도록 정의해야 함
# # -----------------------------------------------------------------------------
# if not hasattr(mp, "get_start_methods"):
#     def _get_start_methods(*, allow_None=False):
#         """
#         DeepSpeed 가 mp.get_start_methods(allow_None=False) 로 호출해도
#         에러 안 나게 키워드 인자를 받아줌.
#         실제 값은 표준 multiprocessing.get_all_start_methods() 기반으로 반환.
#         """
#         print(f"[PATCH] mp.get_start_methods called (allow_None={allow_None})")
#         try:
#             # Python 표준 라이브러리 쪽에서 지원하는 start methods 가져오기
#             if hasattr(py_mp, "get_all_start_methods"):
#                 return py_mp.get_all_start_methods()
#             else:
#                 # 혹시 없으면 일반적인 기본 세트 리턴
#                 return ["fork", "spawn", "forkserver"]
#         except Exception as e:
#             print(f"[PATCH] get_all_start_methods fallback due to {e}")
#             return ["fork", "spawn", "forkserver"]

#     mp.get_start_methods = _get_start_methods
#     print("[PATCH] Added torch.multiprocessing.get_start_methods (with allow_None kwarg)")
# else:
#     print("[DEBUG] mp.has_get_start_methods_at_import: True")


# # -----------------------------------------------------------------------------
# # 🔧 PATCH 1: DataParallelWriterFactory._get_data_parallel_config monkey patch
# #   - UniversalParallelInfo.dp_rank가 없어서 깨지는 버그 우회
# # -----------------------------------------------------------------------------
# import deepspeed.runtime.model_checkpointing.data_parallel_writer_factory as dpwf

# _orig_get_dp_cfg = dpwf.DataParallelWriterFactory._get_data_parallel_config


# def _patched_get_data_parallel_config(self, *args, **kwargs):
#     """
#     DeepSpeed 내부에서 self._uni_parallel_info.dp_rank 접근 전에
#     _uni_parallel_info 인스턴스에 dp_rank 속성을 강제로 꽂아준다.
#     """
#     upi = getattr(self, "_uni_parallel_info", None)
#     if upi is not None and not hasattr(upi, "dp_rank"):
#         if dist.is_initialized():
#             rank = dist.get_rank()
#         else:
#             rank = 0
#         setattr(upi, "dp_rank", rank)
#         print(f"[PATCH] Attached dp_rank={rank} to {type(upi).__name__}")

#     return _orig_get_dp_cfg(self, *args, **kwargs)


# dpwf.DataParallelWriterFactory._get_data_parallel_config = _patched_get_data_parallel_config
# print("[PATCH] Patched DataParallelWriterFactory._get_data_parallel_config to set dp_rank")


# # -----------------------------------------------------------------------------
# # 🔧 PATCH 2: state_dict 안의 텐서를 모두 detach해서 MP 큐에 실어 보내기
# #   - RuntimeError: Cowardly refusing to serialize non-leaf tensor which requires_grad...
# #     에러를 피하기 위한 패치
# #   - 이번에는 새 클래스를 만드는 게 아니라, 원래 클래스의 save() 메서드를 직접 교체
# # -----------------------------------------------------------------------------
# def _detach_tensors(obj):
#     """
#     state_dict 같은 nested 구조 안의 모든 텐서를
#     - autograd graph와 끊고(detach)
#     - 반드시 CPU 텐서로 만들어서
#     - mp.Queue 에 CUDA storage 가 절대 안 실리도록 강제
#     """
#     if isinstance(obj, torch.Tensor):
#         # device 상관없이 전부 CPU로 강제 이동
#         with torch.no_grad():
#             t = obj.detach().cpu().contiguous()
#         return t

#     elif isinstance(obj, dict):
#         return {k: _detach_tensors(v) for k, v in obj.items()}

#     elif isinstance(obj, (list, tuple)):
#         typ = type(obj)
#         return typ(_detach_tensors(v) for v in obj)

#     elif isinstance(obj, set):
#         return {_detach_tensors(v) for v in obj}

#     else:
#         return obj
# # -----------------------------------------------------------------------------
# # 🔧 PATCH 3: ZeRO Stage 1/2 overflow 체크에서 x.float() 전체 캐스팅 때문에 OOM 나는 것 방지
# #   - 기존: grad 전체를 한 번에 float32로 복사해서 isinf/isnan 검사 → +2~3 GiB 피크 메모리
# #   - 변경: grad를 flatten 한 뒤, chunk 단위로만 float() 해서 검사
# # -----------------------------------------------------------------------------
# import deepspeed.runtime.zero.stage_1_and_2 as ds_zero

# _orig_has_inf_or_nan = ds_zero.DeepSpeedZeroOptimizer._has_inf_or_nan

# def _patched_has_inf_or_nan(self, x):
#     """
#     원래 DeepSpeedZeroOptimizer._has_inf_or_nan(self, x)를 대체.
#     - grad 전체를 한 번에 float32로 바꾸지 않고
#     - 작은 chunk 단위로만 float() 호출해서 inf/nan 검사
#     """
#     if x is None:
#         return 0

#     # fp16/bf16일 때만 우리가 직접 검사, 아니면 기존 구현 사용
#     dtype = getattr(self, "dtype", None) or x.dtype
#     if dtype not in (torch.float16, torch.bfloat16):
#         return _orig_has_inf_or_nan(self, x)

#     try:
#         t = x.detach()
#     except Exception as e:
#         print(f"[PATCH_OVERFLOW] detach error: {e}")
#         t = x

#     t = t.contiguous()
#     flat = t.view(-1)

#     # 한 chunk당 요소 개수 (16M 요소 → float32에서 약 64MB 정도)
#     chunk_elems = 16 * 1024 * 1024
#     n = flat.numel()

#     for start in range(0, n, chunk_elems):
#         end = min(start + chunk_elems, n)
#         part = flat[start:end].float()
#         if torch.isinf(part).any() or torch.isnan(part).any():
#             print(f"[PATCH_OVERFLOW] detected inf/nan in grad chunk "
#                   f"(size={end-start}/{n})")
#             return 1

#     return 0

# ds_zero.DeepSpeedZeroOptimizer._has_inf_or_nan = _patched_has_inf_or_nan
# print("[PATCH] Patched DeepSpeedZeroOptimizer._has_inf_or_nan to use chunked float()")

# import deepspeed.runtime.checkpoint_engine.decoupled_checkpoint_engine as dce_mod

# _orig_save = dce_mod.DecoupledCheckpointEngine.save


# def _patched_decoupled_save(self, *args, **kwargs):
#     """
#     DeepSpeed DecoupledCheckpointEngine.save 를 감싸서,
#     Queue 에 넣기 전에 모든 텐서를 detach 하도록 만든 버전.

#     DeepSpeed 쪽에서:
#         save(state_dict=..., path=...)
#     식으로 keyword 인자로 호출하기 때문에, *args, **kwargs 모두 지원.
#     """
#     # --- state_dict, path 뽑기 --- #
#     state = None
#     path = None

#     # 1) positional 로 온 경우 (기본: (state_dict, path))
#     if len(args) >= 1 and isinstance(args[0], dict):
#         state = args[0]
#         if len(args) >= 2:
#             path = args[1]

#     # 2) keyword 로 온 경우
#     if state is None:
#         state = kwargs.get("state_dict", state)
#     if path is None:
#         path = kwargs.get("path", path)

#     if state is None:
#         raise ValueError("[PATCH_SAVE] Could not locate state_dict in args/kwargs")

#     rank = getattr(self, "global_rank", "unknown")
#     print(f"[PATCH_SAVE] rank={rank} path={path} (before detach)")

#     safe_state = _detach_tensors(state)

#     # --- 원래 save 호출을 동일한 시그니처로 재호출 --- #
#     # args 를 다시 구성해서 첫 번째 인자만 safe_state 로 교체
#     if args:
#         new_args = list(args)
#         new_args[0] = safe_state
#         return _orig_save(self, *new_args, **kwargs)
#     else:
#         new_kwargs = dict(kwargs)
#         new_kwargs["state_dict"] = safe_state
#         if path is not None:
#             new_kwargs["path"] = path
#         return _orig_save(self, **new_kwargs)


# dce_mod.DecoupledCheckpointEngine.save = _patched_decoupled_save
# print("[PATCH] Monkey-patched DecoupledCheckpointEngine.save to detach tensors before MP queue")


# # -----------------------------------------------------------------------------
# # Dataset loader
# # -----------------------------------------------------------------------------
# def load_dataset_stream(file_path, tokenizer, block_size=512, max_blocks=None):
#     print(f"[DEBUG] Loading dataset from: {file_path}")
#     blocks = []
#     buffer = []

#     with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#         for line in f:
#             if not line.strip():
#                 continue

#             tokens = tokenizer(line, return_tensors='pt', truncation=False)["input_ids"].squeeze(0)
#             buffer.append(tokens)

#             while sum(b.numel() for b in buffer) >= block_size:
#                 concat = torch.cat(buffer)
#                 blocks.append(concat[:block_size])
#                 buffer = [concat[block_size:]] if concat.numel() > block_size else []

#                 if max_blocks is not None and len(blocks) >= max_blocks:
#                     print(f"[DEBUG] Loaded {len(blocks)} blocks (max_blocks reached).")
#                     return blocks

#     if buffer and (max_blocks is None or len(blocks) < max_blocks):
#         concat = torch.cat(buffer)
#         if concat.numel() >= 1:
#             concat = torch.nn.functional.pad(
#                 concat,
#                 (0, block_size - concat.numel()),
#                 value=tokenizer.pad_token_id,
#             )
#             blocks.append(concat)

#     print(f"[DEBUG] Loaded total {len(blocks)} blocks.")
#     return blocks


# # -----------------------------------------------------------------------------
# # Main training with DeepSpeed + DecoupledCheckpointEngine
# # -----------------------------------------------------------------------------
# def main():
#     # spawn start method 설정
#     # (torch.multiprocessing.get_start_method 는 allow_none 인자를 받음)
#     if mp.get_start_method(allow_none=True) != 'spawn':
#         mp.set_start_method('spawn', force=True)

#     print(f"[DEBUG] mp.has_get_start_methods_in_main: {hasattr(mp, 'get_start_methods')}")

#     # ------------------ Argument Parsing ------------------ #
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--deepspeed_config', type=str, required=True)
#     parser.add_argument('--model_name_or_path', type=str, default='bigscience/bloom-3b')
#     parser.add_argument('--train_file', type=str, required=True)
#     parser.add_argument('--output_dir', type=str, default='./bloom3b-decoupled-finetuned')
#     parser.add_argument('--local_rank', type=int, default=-1)
#     parser.add_argument('--per_device_train_batch_size', type=int, default=2)
#     parser.add_argument('--epochs', type=int, default=1)
#     parser.add_argument('--resume_from', type=str, default=None)
#     args = parser.parse_args()

#     # local_rank 세팅
#     args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
#     if args.local_rank != -1:
#         torch.cuda.set_device(args.local_rank)

#     print(f"[DEBUG] local_rank = {args.local_rank}")
#     print(f"[DEBUG] CUDA available = {torch.cuda.is_available()}")

#     # ------------------ Model and Tokenizer ------------------ #
#     print(f"[DEBUG] Loading model: {args.model_name_or_path}")
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_name_or_path,
#         device_map=None,
#         low_cpu_mem_usage=True,
#         torch_dtype=torch.float16,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

#     # ------------------ Dataset ------------------ #
#     train_blocks = load_dataset_stream(
#         args.train_file,
#         tokenizer,
#         block_size=512,
#         max_blocks=450,
#     )

#     # ------------------ DeepSpeed Engine Initialization ------------------ #
#     print("[DEBUG] Initializing DeepSpeed engine...")
#     ds_engine, optimizer, _, _ = deepspeed.initialize(
#         args=args,
#         model=model,
#         model_parameters=model.parameters(),
#     )

#     global_rank = ds_engine.global_rank
#     print(f"[DEBUG] DeepSpeed config (rank {global_rank}): {ds_engine.config}")

#     # ------------------ Checkpoint Engine Print Statement ------------------ #
#     if ds_engine.checkpoint_engine:
#         print(f"[Rank {global_rank}] Using checkpoint engine: {type(ds_engine.checkpoint_engine).__name__}")
#     else:
#         print(f"[Rank {global_rank}] No checkpoint engine detected. Using default behavior.")

#     model.train()

#     ckpt_dir = os.path.join(args.output_dir, "checkpoint-1")

#     save_interval = 500
#     latest_epoch = 0

#     for epoch in range(latest_epoch, args.epochs):
#         print(f"[Rank {global_rank}] Entering epoch {epoch+1}")

#         for i, block in enumerate(train_blocks):
#             inputs = block.unsqueeze(0).to(ds_engine.device)
#             labels = inputs.clone()

#             outputs = ds_engine(inputs, labels=labels)
#             loss = outputs.loss
#             ds_engine.backward(loss)

#             ds_engine.step()

#             if i % 50 == 0 and global_rank == 0:
#                 print(f"[Rank 0] Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

#             checkpoint_now = (i % 50 == 0) or ((i + 1) % save_interval == 0) or (i + 1 == len(train_blocks))
#             if checkpoint_now:
#                 start_time = time.time()
#                 print(f"[Rank {global_rank}] Starting save_checkpoint at step {i}...")
#                 ds_engine.save_checkpoint(save_dir=ckpt_dir, tag=f"my_decoupled_ckpt_e{epoch+1}_s{i}")
#                 init_duration = time.time() - start_time
#                 print(f"[Rank {global_rank}] Time to initiate save: {init_duration:.2f} seconds")

#                 if ds_engine.checkpoint_engine:
#                     ds_engine.checkpoint_engine.cleanup()
#                 total_duration = time.time() - start_time
#                 print(f"[Rank {global_rank}] Total time to save and cleanup: {total_duration:.2f} seconds")


# if __name__ == "__main__":
#     print("[DEBUG] >>> __main__ entered, calling main()")
#     main()

# import os
# import time
# import argparse
# import csv
# from datetime import datetime

# import torch
# import torch.multiprocessing as mp
# import torch.distributed as dist
# import deepspeed
# import multiprocessing as py_mp  # ★ 표준 multiprocessing

# print("[DEBUG] >>> decouple_train_bloom3b.py imported")

# from transformers import AutoModelForCausalLM, AutoTokenizer
# from deepspeed.runtime.checkpoint_engine.decoupled_checkpoint_engine import DecoupledCheckpointEngine

# # -----------------------------------------------------------------------------
# # 🔧 PATCH 0: torch.multiprocessing.get_start_methods 없을 때 추가
# #   - DeepSpeed가 mp.get_start_methods(allow_None=False) 를 호출해서,
# #     시그니처가 (allow_None=...) 를 받도록 정의해야 함
# # -----------------------------------------------------------------------------
# if not hasattr(mp, "get_start_methods"):
#     def _get_start_methods(*, allow_None=False):
#         """
#         DeepSpeed 가 mp.get_start_methods(allow_None=False) 로 호출해도
#         에러 안 나게 키워드 인자를 받아줌.
#         실제 값은 표준 multiprocessing.get_all_start_methods() 기반으로 반환.
#         """
#         print(f"[PATCH] mp.get_start_methods called (allow_None={allow_None})")
#         try:
#             if hasattr(py_mp, "get_all_start_methods"):
#                 return py_mp.get_all_start_methods()
#             else:
#                 return ["fork", "spawn", "forkserver"]
#         except Exception as e:
#             print(f"[PATCH] get_all_start_methods fallback due to {e}")
#             return ["fork", "spawn", "forkserver"]

#     mp.get_start_methods = _get_start_methods
#     print("[PATCH] Added torch.multiprocessing.get_start_methods (with allow_None kwarg)")
# else:
#     print("[DEBUG] mp.has_get_start_methods_at_import: True")


# # -----------------------------------------------------------------------------
# # 🔧 PATCH 1: DataParallelWriterFactory._get_data_parallel_config monkey patch
# #   - UniversalParallelInfo.dp_rank가 없어서 깨지는 버그 우회
# # -----------------------------------------------------------------------------
# import deepspeed.runtime.model_checkpointing.data_parallel_writer_factory as dpwf

# _orig_get_dp_cfg = dpwf.DataParallelWriterFactory._get_data_parallel_config


# def _patched_get_data_parallel_config(self, *args, **kwargs):
#     """
#     DeepSpeed 내부에서 self._uni_parallel_info.dp_rank 접근 전에
#     _uni_parallel_info 인스턴스에 dp_rank 속성을 강제로 꽂아준다.
#     """
#     upi = getattr(self, "_uni_parallel_info", None)
#     if upi is not None and not hasattr(upi, "dp_rank"):
#         if dist.is_initialized():
#             rank = dist.get_rank()
#         else:
#             rank = 0
#         setattr(upi, "dp_rank", rank)
#         print(f"[PATCH] Attached dp_rank={rank} to {type(upi).__name__}")

#     return _orig_get_dp_cfg(self, *args, **kwargs)


# dpwf.DataParallelWriterFactory._get_data_parallel_config = _patched_get_data_parallel_config
# print("[PATCH] Patched DataParallelWriterFactory._get_data_parallel_config to set dp_rank")


# # -----------------------------------------------------------------------------
# # 🔧 PATCH 2: ZeRO Stage 1/2 overflow 체크 OOM 방지용 chunked float()
# # -----------------------------------------------------------------------------
# import deepspeed.runtime.zero.stage_1_and_2 as ds_zero

# _orig_has_inf_or_nan = ds_zero.DeepSpeedZeroOptimizer._has_inf_or_nan


# def _patched_has_inf_or_nan(self, x):
#     """
#     원래 DeepSpeedZeroOptimizer._has_inf_or_nan(self, x)를 대체.
#     - grad 전체를 한 번에 float32로 바꾸지 않고
#     - 작은 chunk 단위로만 float() 호출해서 inf/nan 검사
#     """
#     if x is None:
#         return 0

#     dtype = getattr(self, "dtype", None) or x.dtype
#     if dtype not in (torch.float16, torch.bfloat16):
#         return _orig_has_inf_or_nan(self, x)

#     try:
#         t = x.detach()
#     except Exception as e:
#         print(f"[PATCH_OVERFLOW] detach error: {e}")
#         t = x

#     t = t.contiguous()
#     flat = t.view(-1)

#     # 한 chunk당 요소 개수 (16M 요소 → float32에서 약 64MB 정도)
#     chunk_elems = 16 * 1024 * 1024
#     n = flat.numel()

#     for start in range(0, n, chunk_elems):
#         end = min(start + chunk_elems, n)
#         part = flat[start:end].float()
#         if torch.isinf(part).any() or torch.isnan(part).any():
#             print(f"[PATCH_OVERFLOW] detected inf/nan in grad chunk "
#                   f"(size={end-start}/{n})")
#             return 1

#     return 0


# ds_zero.DeepSpeedZeroOptimizer._has_inf_or_nan = _patched_has_inf_or_nan
# print("[PATCH] Patched DeepSpeedZeroOptimizer._has_inf_or_nan to use chunked float()")


# # -----------------------------------------------------------------------------
# # 🔧 PATCH 3: state_dict 안 텐서를 detach + CPU로 옮기고,
# #            checkpoint I/O 통계를 last_timing 으로 노출
# # -----------------------------------------------------------------------------
# def _detach_tensors(obj):
#     """
#     state_dict 같은 nested 구조 안의 모든 텐서를
#     - autograd graph와 끊고(detach)
#     - 반드시 CPU 텐서로 만들어서
#     - mp.Queue 에 CUDA storage 가 절대 안 실리도록 강제
#     """
#     if isinstance(obj, torch.Tensor):
#         with torch.no_grad():
#             t = obj.detach().cpu().contiguous()
#         return t

#     elif isinstance(obj, dict):
#         return {k: _detach_tensors(v) for k, v in obj.items()}

#     elif isinstance(obj, (list, tuple)):
#         typ = type(obj)
#         return typ(_detach_tensors(v) for v in obj)

#     elif isinstance(obj, set):
#         return {_detach_tensors(v) for v in obj}

#     else:
#         return obj


# def _tensor_stats(obj):
#     """
#     nested 구조의 텐서들에 대해
#     - 전체 element 수
#     - 전체 logical bytes (numel * element_size)
#     를 계산.
#     """
#     if isinstance(obj, torch.Tensor):
#         return obj.numel(), obj.numel() * obj.element_size()

#     elif isinstance(obj, dict):
#         n = 0
#         b = 0
#         for v in obj.values():
#             nn, bb = _tensor_stats(v)
#             n += nn
#             b += bb
#         return n, b

#     elif isinstance(obj, (list, tuple, set)):
#         n = 0
#         b = 0
#         for v in obj:
#             nn, bb = _tensor_stats(v)
#             n += nn
#             b += bb
#         return n, b

#     else:
#         return 0, 0


# import deepspeed.runtime.checkpoint_engine.decoupled_checkpoint_engine as dce_mod

# _orig_save = dce_mod.DecoupledCheckpointEngine.save


# def _patched_decoupled_save(self, *args, **kwargs):
#     """
#     DecoupledCheckpointEngine.save 를 감싸서
#     - Queue 에 넣기 전에 모든 텐서를 detach + CPU로 이동
#     - 경로(path)를 보고 model/optimizer 구분
#     - I/O 통계를 self.last_timing 에 저장
#     """
#     # --- state_dict, path 뽑기 --- #
#     state = None
#     path = None

#     if len(args) >= 1 and isinstance(args[0], dict):
#         state = args[0]
#         if len(args) >= 2:
#             path = args[1]

#     if state is None:
#         state = kwargs.get("state_dict", state)
#     if path is None:
#         path = kwargs.get("path", path)

#     if state is None:
#         raise ValueError("[PATCH_SAVE] Could not locate state_dict in args/kwargs")

#     rank = getattr(self, "global_rank", "unknown")
#     print(f"[PATCH_SAVE] rank={rank} path={path} (before detach)")

#     safe_state = _detach_tensors(state)

#     # ---- 실제 저장 호출 ---- #
#     start = time.time()
#     if args:
#         new_args = list(args)
#         new_args[0] = safe_state
#         result = _orig_save(self, *new_args, **kwargs)
#     else:
#         new_kwargs = dict(kwargs)
#         new_kwargs["state_dict"] = safe_state
#         if path is not None:
#             new_kwargs["path"] = path
#         result = _orig_save(self, **new_kwargs)
#     end = time.time()
#     duration = end - start

#     # ---- 통계 계산 ---- #
#     numel, logical_bytes = _tensor_stats(safe_state)
#     kind = "model"
#     if path is not None and "optim_states" in str(path):
#         kind = "optim"

#     if not hasattr(self, "last_timing") or not isinstance(self.last_timing, dict):
#         self.last_timing = {}

#     prefix = "model" if kind == "model" else "optim"

#     self.last_timing[f"{prefix}_tensor_elements"] = int(numel)
#     self.last_timing[f"{prefix}_logical_bytes"] = int(logical_bytes)
#     self.last_timing[f"{prefix}_cpu2disk_seconds"] = float(duration)

#     if duration > 0 and logical_bytes > 0:
#         self.last_timing[f"{prefix}_throughput_GBps"] = float(
#             logical_bytes / (duration * (1024 ** 3))
#         )
#     else:
#         self.last_timing[f"{prefix}_throughput_GBps"] = 0.0

#     # GPU→CPU, enqueue, flush, barrier 는 여기서 직접 쪼개기 힘들어서 0으로 둠
#     self.last_timing.setdefault(f"{prefix}_gpu2cpu_seconds", 0.0)
#     self.last_timing.setdefault("enqueue_seconds", 0.0)
#     self.last_timing.setdefault("flush_seconds", 0.0)
#     self.last_timing.setdefault("barrier_seconds", 0.0)

#     return result


# dce_mod.DecoupledCheckpointEngine.save = _patched_decoupled_save
# print("[PATCH] Monkey-patched DecoupledCheckpointEngine.save to detach tensors & record stats")


# # -----------------------------------------------------------------------------
# # Dataset loader
# # -----------------------------------------------------------------------------
# def load_dataset_stream(file_path, tokenizer, block_size=512, max_blocks=None):
#     print(f"[DEBUG] Loading dataset from: {file_path}")
#     blocks = []
#     buffer = []

#     with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#         for line in f:
#             if not line.strip():
#                 continue

#             tokens = tokenizer(line, return_tensors='pt', truncation=False)["input_ids"].squeeze(0)
#             buffer.append(tokens)

#             while sum(b.numel() for b in buffer) >= block_size:
#                 concat = torch.cat(buffer)
#                 blocks.append(concat[:block_size])
#                 buffer = [concat[block_size:]] if concat.numel() > block_size else []

#                 if max_blocks is not None and len(blocks) >= max_blocks:
#                     print(f"[DEBUG] Loaded {len(blocks)} blocks (max_blocks reached).")
#                     return blocks

#     if buffer and (max_blocks is None or len(blocks) < max_blocks):
#         concat = torch.cat(buffer)
#         if concat.numel() >= 1:
#             concat = torch.nn.functional.pad(
#                 concat,
#                 (0, block_size - concat.numel()),
#                 value=tokenizer.pad_token_id,
#             )
#             blocks.append(concat)

#     print(f"[DEBUG] Loaded total {len(blocks)} blocks.")
#     return blocks


# # -----------------------------------------------------------------------------
# # CSV Logger
# # -----------------------------------------------------------------------------
# def create_csv_logger(output_dir, rank):
#     os.makedirs(output_dir, exist_ok=True)
#     log_path = os.path.join(output_dir, f"decoupled_timing_rank{rank}.csv")
#     is_new = not os.path.exists(log_path)

#     f = open(log_path, "a", newline="")
#     writer = csv.writer(f)

#     if is_new:
#         writer.writerow([
#             "timestamp",
#             "epoch",
#             "step",
#             "loss",
#             "rank",
#             "has_checkpoint",
#             "total_ckpt_seconds",
#             "enqueue_seconds",
#             "flush_seconds",
#             "barrier_seconds",
#             "model_gpu2cpu_seconds",
#             "model_cpu2disk_seconds",
#             "optim_gpu2cpu_seconds",
#             "optim_cpu2disk_seconds",
#             "model_tensor_elements",
#             "optim_tensor_elements",
#             "model_logical_bytes",
#             "optim_logical_bytes",
#             "model_throughput_GBps",
#             "optim_throughput_GBps",
#             "backward_seconds",
#             "step_seconds",
#         ])

#     return f, writer


# # -----------------------------------------------------------------------------
# # Main training with DeepSpeed + DecoupledCheckpointEngine
# # -----------------------------------------------------------------------------
# def main():
#     # spawn start method 설정
#     if mp.get_start_method(allow_none=True) != 'spawn':
#         mp.set_start_method('spawn', force=True)

#     print(f"[DEBUG] mp.has_get_start_methods_in_main: {hasattr(mp, 'get_start_methods')}")

#     # ------------------ Argument Parsing ------------------ #
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--deepspeed_config', type=str, required=True)
#     parser.add_argument('--model_name_or_path', type=str, default='bigscience/bloom-3b')
#     parser.add_argument('--train_file', type=str, required=True)
#     parser.add_argument('--output_dir', type=str, default='./bloom3b-decoupled-finetuned')
#     parser.add_argument('--local_rank', type=int, default=-1)
#     parser.add_argument('--per_device_train_batch_size', type=int, default=2)
#     parser.add_argument('--epochs', type=int, default=3)  # 여러 epoch 돌리면 step 수↑
#     parser.add_argument('--resume_from', type=str, default=None)
#     parser.add_argument('--max_steps', type=int, default=1070)            # ★ 목표 step 수
#     parser.add_argument('--checkpoint_interval', type=int, default=200)   # ★ 200마다 ckpt
#     args = parser.parse_args()

#     # local_rank 세팅
#     args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
#     if args.local_rank != -1:
#         torch.cuda.set_device(args.local_rank)

#     print(f"[DEBUG] local_rank = {args.local_rank}")
#     print(f"[DEBUG] CUDA available = {torch.cuda.is_available()}")

#     # ------------------ Model and Tokenizer ------------------ #
#     print(f"[DEBUG] Loading model: {args.model_name_or_path}")
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_name_or_path,
#         device_map=None,
#         low_cpu_mem_usage=True,
#         torch_dtype=torch.float16,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

#     # ------------------ Dataset ------------------ #
#     train_blocks = load_dataset_stream(
#         args.train_file,
#         tokenizer,
#         block_size=512,
#         max_blocks=450,  # 필요하면 늘리거나 None으로
#     )

#     # ------------------ DeepSpeed Engine Initialization ------------------ #
#     print("[DEBUG] Initializing DeepSpeed engine...")
#     ds_engine, optimizer, _, _ = deepspeed.initialize(
#         args=args,
#         model=model,
#         model_parameters=model.parameters(),
#     )

#     global_rank = ds_engine.global_rank
#     print(f"[DEBUG] DeepSpeed config (rank {global_rank}): {ds_engine.config}")

#     if ds_engine.checkpoint_engine:
#         print(f"[Rank {global_rank}] Using checkpoint engine: {type(ds_engine.checkpoint_engine).__name__}")
#     else:
#         print(f"[Rank {global_rank}] No checkpoint engine detected. Using default behavior.")

#     # ------------------ CSV Logger ------------------ #
#     log_f, log_writer = create_csv_logger(args.output_dir, global_rank)
#     print(f"[DEBUG] Rank {global_rank} CSV log at {log_f.name}", flush=True)

#     model.train()

#     max_steps = args.max_steps
#     ckpt_interval = args.checkpoint_interval

#     step = 0  # global step
#     for epoch in range(args.epochs):
#         print(f"[Rank {global_rank}] Entering epoch {epoch+1}")

#         for i, block in enumerate(train_blocks):
#             if step > max_steps:
#                 break

#             # ---- step 타이밍 시작 ----
#             step_start = time.time()

#             inputs = block.unsqueeze(0).to(ds_engine.device)
#             labels = inputs.clone()

#             # forward
#             outputs = ds_engine(inputs, labels=labels)
#             loss = outputs.loss

#             if global_rank == 0:
#                 print(f"[Rank 0] Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

#             # backward timing
#             bwd_start = time.time()
#             ds_engine.backward(loss)
#             bwd_end = time.time()
#             backward_seconds = bwd_end - bwd_start

#             # optimizer step
#             ds_engine.step()
#             step_end = time.time()
#             step_seconds = step_end - step_start

#             # ---- checkpoint 여부 결정 ----
#             checkpoint_now = (step % ckpt_interval == 0) or (step == max_steps)

#             has_checkpoint = 0
#             total_ckpt_seconds = 0.0
#             enqueue_seconds = 0.0
#             flush_seconds = 0.0
#             barrier_seconds = 0.0
#             model_gpu2cpu_seconds = 0.0
#             model_cpu2disk_seconds = 0.0
#             optim_gpu2cpu_seconds = 0.0
#             optim_cpu2disk_seconds = 0.0
#             model_tensor_elements = 0
#             optim_tensor_elements = 0
#             model_logical_bytes = 0
#             optim_logical_bytes = 0
#             model_throughput_GBps = 0.0
#             optim_throughput_GBps = 0.0

#             if checkpoint_now:
#                 has_checkpoint = 1
#                 ckpt_dir = os.path.join(args.output_dir, f"checkpoint-1")

#                 tag = f"my_decoupled_ckpt_e{epoch+1}_s{step}"

#                 if global_rank == 0:
#                     print(f"[Rank {global_rank}] Starting save_checkpoint at step {step}...")

#                 t_ckpt_start = time.time()
#                 ds_engine.save_checkpoint(save_dir=ckpt_dir, tag=tag)
#                 # 필요하면 cleanup 호출
#                 if ds_engine.checkpoint_engine:
#                     ds_engine.checkpoint_engine.cleanup()
#                 t_ckpt_end = time.time()
#                 total_ckpt_seconds = t_ckpt_end - t_ckpt_start

#                 # 패치에서 기록해둔 timing/stat 가져오기
#                 stats = getattr(ds_engine.checkpoint_engine, "last_timing", {}) if ds_engine.checkpoint_engine else {}

#                 enqueue_seconds = float(stats.get("enqueue_seconds", 0.0))
#                 flush_seconds = float(stats.get("flush_seconds", 0.0))
#                 barrier_seconds = float(stats.get("barrier_seconds", 0.0))

#                 model_gpu2cpu_seconds = float(stats.get("model_gpu2cpu_seconds", 0.0))
#                 model_cpu2disk_seconds = float(stats.get("model_cpu2disk_seconds", 0.0))
#                 optim_gpu2cpu_seconds = float(stats.get("optim_gpu2cpu_seconds", 0.0))
#                 optim_cpu2disk_seconds = float(stats.get("optim_cpu2disk_seconds", 0.0))

#                 model_tensor_elements = int(stats.get("model_tensor_elements", 0))
#                 optim_tensor_elements = int(stats.get("optim_tensor_elements", 0))
#                 model_logical_bytes = int(stats.get("model_logical_bytes", 0))
#                 optim_logical_bytes = int(stats.get("optim_logical_bytes", 0))
#                 model_throughput_GBps = float(stats.get("model_throughput_GBps", 0.0))
#                 optim_throughput_GBps = float(stats.get("optim_throughput_GBps", 0.0))

#                 if global_rank == 0:
#                     print(f"[Rank {global_rank}] Total time to save and cleanup: {total_ckpt_seconds:.2f} seconds")

#             # ---- CSV 로그 한 줄 쓰기 ----
#             ts = datetime.utcnow().isoformat()

#             log_row = [
#                 ts,
#                 epoch + 1,
#                 step,
#                 float(loss.item()),
#                 global_rank,
#                 has_checkpoint,
#                 total_ckpt_seconds,
#                 enqueue_seconds,
#                 flush_seconds,
#                 barrier_seconds,
#                 model_gpu2cpu_seconds,
#                 model_cpu2disk_seconds,
#                 optim_gpu2cpu_seconds,
#                 optim_cpu2disk_seconds,
#                 model_tensor_elements,
#                 optim_tensor_elements,
#                 model_logical_bytes,
#                 optim_logical_bytes,
#                 model_throughput_GBps,
#                 optim_throughput_GBps,
#                 backward_seconds,
#                 step_seconds,
#             ]

#             log_writer.writerow(log_row)
#             log_f.flush()

#             step += 1

#         if step > max_steps:
#             break

#     log_f.close()
#     print(f"[Rank {global_rank}] Training finished. CSV log closed.", flush=True)


# if __name__ == "__main__":
#     print("[DEBUG] >>> __main__ entered, calling main()")
#     main()


# import os
# import time
# import argparse
# import csv
# from datetime import datetime

# import torch
# import torch.multiprocessing as mp
# import torch.distributed as dist
# import deepspeed
# import multiprocessing as py_mp  # ★ 표준 multiprocessing

# print("[DEBUG] >>> decouple_train_bloom3b.py imported")

# from transformers import AutoModelForCausalLM, AutoTokenizer
# from deepspeed.runtime.checkpoint_engine.decoupled_checkpoint_engine import DecoupledCheckpointEngine

# # -----------------------------------------------------------------------------
# # 🔧 PATCH 0: torch.multiprocessing.get_start_methods 없을 때 추가
# #   - DeepSpeed가 mp.get_start_methods(allow_None=False) 를 호출해서,
# #     시그니처가 (allow_None=...) 를 받도록 정의해야 함
# # -----------------------------------------------------------------------------
# if not hasattr(mp, "get_start_methods"):
#     def _get_start_methods(*, allow_None=False):
#         """
#         DeepSpeed 가 mp.get_start_methods(allow_None=False) 로 호출해도
#         에러 안 나게 키워드 인자를 받아줌.
#         실제 값은 표준 multiprocessing.get_all_start_methods() 기반으로 반환.
#         """
#         print(f"[PATCH] mp.get_start_methods called (allow_None={allow_None})")
#         try:
#             if hasattr(py_mp, "get_all_start_methods"):
#                 return py_mp.get_all_start_methods()
#             else:
#                 return ["fork", "spawn", "forkserver"]
#         except Exception as e:
#             print(f"[PATCH] get_all_start_methods fallback due to {e}")
#             return ["fork", "spawn", "forkserver"]

#     mp.get_start_methods = _get_start_methods
#     print("[PATCH] Added torch.multiprocessing.get_start_methods (with allow_None kwarg)")
# else:
#     print("[DEBUG] mp.has_get_start_methods_at_import: True")


# # -----------------------------------------------------------------------------
# # 🔧 PATCH 1: DataParallelWriterFactory._get_data_parallel_config monkey patch
# #   - UniversalParallelInfo.dp_rank가 없어서 깨지는 버그 우회
# # -----------------------------------------------------------------------------
# import deepspeed.runtime.model_checkpointing.data_parallel_writer_factory as dpwf

# _orig_get_dp_cfg = dpwf.DataParallelWriterFactory._get_data_parallel_config


# def _patched_get_data_parallel_config(self, *args, **kwargs):
#     """
#     DeepSpeed 내부에서 self._uni_parallel_info.dp_rank 접근 전에
#     _uni_parallel_info 인스턴스에 dp_rank 속성을 강제로 꽂아준다.
#     """
#     upi = getattr(self, "_uni_parallel_info", None)
#     if upi is not None and not hasattr(upi, "dp_rank"):
#         if dist.is_initialized():
#             rank = dist.get_rank()
#         else:
#             rank = 0
#         setattr(upi, "dp_rank", rank)
#         print(f"[PATCH] Attached dp_rank={rank} to {type(upi).__name__}")

#     return _orig_get_dp_cfg(self, *args, **kwargs)


# dpwf.DataParallelWriterFactory._get_data_parallel_config = _patched_get_data_parallel_config
# print("[PATCH] Patched DataParallelWriterFactory._get_data_parallel_config to set dp_rank")


# # -----------------------------------------------------------------------------
# # 🔧 PATCH 2: ZeRO Stage 1/2 overflow 체크 OOM 방지용 chunked float()
# # -----------------------------------------------------------------------------
# import deepspeed.runtime.zero.stage_1_and_2 as ds_zero

# _orig_has_inf_or_nan = ds_zero.DeepSpeedZeroOptimizer._has_inf_or_nan


# def _patched_has_inf_or_nan(self, x):
#     """
#     원래 DeepSpeedZeroOptimizer._has_inf_or_nan(self, x)를 대체.
#     - grad 전체를 한 번에 float32로 바꾸지 않고
#     - 작은 chunk 단위로만 float() 호출해서 inf/nan 검사
#     """
#     if x is None:
#         return 0

#     dtype = getattr(self, "dtype", None) or x.dtype
#     if dtype not in (torch.float16, torch.bfloat16):
#         return _orig_has_inf_or_nan(self, x)

#     try:
#         t = x.detach()
#     except Exception as e:
#         print(f"[PATCH_OVERFLOW] detach error: {e}")
#         t = x

#     t = t.contiguous()
#     flat = t.view(-1)

#     # 한 chunk당 요소 개수 (16M 요소 → float32에서 약 64MB 정도)
#     chunk_elems = 16 * 1024 * 1024
#     n = flat.numel()

#     for start in range(0, n, chunk_elems):
#         end = min(start + chunk_elems, n)
#         part = flat[start:end].float()
#         if torch.isinf(part).any() or torch.isnan(part).any():
#             print(f"[PATCH_OVERFLOW] detected inf/nan in grad chunk "
#                   f"(size={end-start}/{n})")
#             return 1

#     return 0


# ds_zero.DeepSpeedZeroOptimizer._has_inf_or_nan = _patched_has_inf_or_nan
# print("[PATCH] Patched DeepSpeedZeroOptimizer._has_inf_or_nan to use chunked float()")


# # -----------------------------------------------------------------------------
# # 🔧 PATCH 3: state_dict 안 텐서를 detach + CPU로 옮기고,
# #            checkpoint I/O 통계를 last_timing 으로 노출
# # -----------------------------------------------------------------------------
# def _detach_tensors(obj):
#     """
#     state_dict 같은 nested 구조 안의 모든 텐서를
#     - autograd graph와 끊고(detach)
#     - 반드시 CPU 텐서로 만들어서
#     - mp.Queue 에 CUDA storage 가 절대 안 실리도록 강제
#     """
#     if isinstance(obj, torch.Tensor):
#         with torch.no_grad():
#             t = obj.detach().cpu().contiguous()
#         return t

#     elif isinstance(obj, dict):
#         return {k: _detach_tensors(v) for k, v in obj.items()}

#     elif isinstance(obj, (list, tuple)):
#         typ = type(obj)
#         return typ(_detach_tensors(v) for v in obj)

#     elif isinstance(obj, set):
#         return {_detach_tensors(v) for v in obj}

#     else:
#         return obj


# def _tensor_stats(obj):
#     """
#     nested 구조의 텐서들에 대해
#     - 전체 element 수
#     - 전체 logical bytes (numel * element_size)
#     를 계산.
#     """
#     if isinstance(obj, torch.Tensor):
#         return obj.numel(), obj.numel() * obj.element_size()

#     elif isinstance(obj, dict):
#         n = 0
#         b = 0
#         for v in obj.values():
#             nn, bb = _tensor_stats(v)
#             n += nn
#             b += bb
#         return n, b

#     elif isinstance(obj, (list, tuple, set)):
#         n = 0
#         b = 0
#         for v in obj:
#             nn, bb = _tensor_stats(v)
#             n += nn
#             b += bb
#         return n, b

#     else:
#         return 0, 0


# import deepspeed.runtime.checkpoint_engine.decoupled_checkpoint_engine as dce_mod

# _orig_save = dce_mod.DecoupledCheckpointEngine.save


# def _patched_decoupled_save(self, *args, **kwargs):
#     """
#     DecoupledCheckpointEngine.save 를 감싸서
#     - Queue 에 넣기 전에 모든 텐서를 detach + CPU로 이동
#     - 경로(path)를 보고 model/optimizer 구분
#     - I/O 통계를 self.last_timing 에 저장
#     """
#     # --- state_dict, path 뽑기 --- #
#     state = None
#     path = None

#     if len(args) >= 1 and isinstance(args[0], dict):
#         state = args[0]
#         if len(args) >= 2:
#             path = args[1]

#     if state is None:
#         state = kwargs.get("state_dict", state)
#     if path is None:
#         path = kwargs.get("path", path)

#     if state is None:
#         raise ValueError("[PATCH_SAVE] Could not locate state_dict in args/kwargs")

#     rank = getattr(self, "global_rank", "unknown")
#     print(f"[PATCH_SAVE] rank={rank} path={path} (before detach)")

#     safe_state = _detach_tensors(state)

#     # ---- 실제 저장 호출 ---- #
#     start = time.time()
#     if args:
#         new_args = list(args)
#         new_args[0] = safe_state
#         result = _orig_save(self, *new_args, **kwargs)
#     else:
#         new_kwargs = dict(kwargs)
#         new_kwargs["state_dict"] = safe_state
#         if path is not None:
#             new_kwargs["path"] = path
#         result = _orig_save(self, **new_kwargs)
#     end = time.time()
#     duration = end - start

#     # ---- 통계 계산 ---- #
#     numel, logical_bytes = _tensor_stats(safe_state)
#     kind = "model"
#     if path is not None and "optim_states" in str(path):
#         kind = "optim"

#     if not hasattr(self, "last_timing") or not isinstance(self.last_timing, dict):
#         self.last_timing = {}

#     prefix = "model" if kind == "model" else "optim"

#     self.last_timing[f"{prefix}_tensor_elements"] = int(numel)
#     self.last_timing[f"{prefix}_logical_bytes"] = int(logical_bytes)
#     self.last_timing[f"{prefix}_cpu2disk_seconds"] = float(duration)

#     if duration > 0 and logical_bytes > 0:
#         self.last_timing[f"{prefix}_throughput_GBps"] = float(
#             logical_bytes / (duration * (1024 ** 3))
#         )
#     else:
#         self.last_timing[f"{prefix}_throughput_GBps"] = 0.0

#     # GPU→CPU, enqueue, flush, barrier 는 여기서 직접 쪼개기 힘들어서 0으로 둠
#     self.last_timing.setdefault(f"{prefix}_gpu2cpu_seconds", 0.0)
#     self.last_timing.setdefault("enqueue_seconds", 0.0)
#     self.last_timing.setdefault("flush_seconds", 0.0)
#     self.last_timing.setdefault("barrier_seconds", 0.0)

#     return result


# dce_mod.DecoupledCheckpointEngine.save = _patched_decoupled_save
# print("[PATCH] Monkey-patched DecoupledCheckpointEngine.save to detach tensors & record stats")


# # -----------------------------------------------------------------------------
# # Dataset loader
# # -----------------------------------------------------------------------------
# def load_dataset_stream(file_path, tokenizer, block_size=512, max_blocks=None):
#     print(f"[DEBUG] Loading dataset from: {file_path}")
#     blocks = []
#     buffer = []

#     with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#         for line in f:
#             if not line.strip():
#                 continue

#             tokens = tokenizer(line, return_tensors='pt', truncation=False)["input_ids"].squeeze(0)
#             buffer.append(tokens)

#             while sum(b.numel() for b in buffer) >= block_size:
#                 concat = torch.cat(buffer)
#                 blocks.append(concat[:block_size])
#                 buffer = [concat[block_size:]] if concat.numel() > block_size else []

#                 if max_blocks is not None and len(blocks) >= max_blocks:
#                     print(f"[DEBUG] Loaded {len(blocks)} blocks (max_blocks reached).")
#                     return blocks

#     if buffer and (max_blocks is None or len(blocks) < max_blocks):
#         concat = torch.cat(buffer)
#         if concat.numel() >= 1:
#             concat = torch.nn.functional.pad(
#                 concat,
#                 (0, block_size - concat.numel()),
#                 value=tokenizer.pad_token_id,
#             )
#             blocks.append(concat)

#     print(f"[DEBUG] Loaded total {len(blocks)} blocks.")
#     return blocks


# # -----------------------------------------------------------------------------
# # CSV Logger
# # -----------------------------------------------------------------------------
# def create_csv_logger(output_dir, rank):
#     os.makedirs(output_dir, exist_ok=True)

#     # 🔴 수정: rank 0만 CSV를 쓴다. 나머지 rank는 로그를 남기지 않음.
#     if rank != 0:
#         print(f"[DEBUG] Rank {rank} will NOT create CSV logger (only rank 0 logs).")
#         return None, None

#     log_path = os.path.join(output_dir, f"decoupled_timing_rank{rank}.csv")
#     is_new = not os.path.exists(log_path)

#     f = open(log_path, "a", newline="")
#     writer = csv.writer(f)

#     if is_new:
#         writer.writerow([
#             "timestamp",
#             "epoch",
#             "step",
#             "loss",
#             "rank",
#             "has_checkpoint",
#             "total_ckpt_seconds",
#             "enqueue_seconds",
#             "flush_seconds",
#             "barrier_seconds",
#             "model_gpu2cpu_seconds",
#             "model_cpu2disk_seconds",
#             "optim_gpu2cpu_seconds",
#             "optim_cpu2disk_seconds",
#             "model_tensor_elements",
#             "optim_tensor_elements",
#             "model_logical_bytes",
#             "optim_logical_bytes",
#             "model_throughput_GBps",
#             "optim_throughput_GBps",
#             "backward_seconds",
#             "step_seconds",
#         ])

#     print(f"[DEBUG] Rank {rank} CSV logger created at {log_path}")
#     return f, writer


# # -----------------------------------------------------------------------------
# # Main training with DeepSpeed + DecoupledCheckpointEngine
# # -----------------------------------------------------------------------------
# def main():
#     # spawn start method 설정
#     if mp.get_start_method(allow_none=True) != 'spawn':
#         mp.set_start_method('spawn', force=True)

#     print(f"[DEBUG] mp.has_get_start_methods_in_main: {hasattr(mp, 'get_start_methods')}")

#     # ------------------ Argument Parsing ------------------ #
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--deepspeed_config', type=str, required=True)
#     parser.add_argument('--model_name_or_path', type=str, default='bigscience/bloom-3b')
#     parser.add_argument('--train_file', type=str, required=True)
#     parser.add_argument('--output_dir', type=str, default='./bloom3b-decoupled-finetuned')
#     parser.add_argument('--local_rank', type=int, default=-1)
#     parser.add_argument('--per_device_train_batch_size', type=int, default=2)
#     parser.add_argument('--epochs', type=int, default=3)  # 여러 epoch 돌리면 step 수↑
#     parser.add_argument('--resume_from', type=str, default=None)
#     parser.add_argument('--max_steps', type=int, default=1070)            # ★ 목표 step 수
#     parser.add_argument('--checkpoint_interval', type=int, default=200)   # ★ 200마다 ckpt
#     args = parser.parse_args()

#     # local_rank 세팅
#     args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
#     if args.local_rank != -1:
#         torch.cuda.set_device(args.local_rank)

#     print(f"[DEBUG] local_rank = {args.local_rank}")
#     print(f"[DEBUG] CUDA available = {torch.cuda.is_available()}")

#     # ------------------ Model and Tokenizer ------------------ #
#     print(f"[DEBUG] Loading model: {args.model_name_or_path}")
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_name_or_path,
#         device_map=None,
#         low_cpu_mem_usage=True,
#         torch_dtype=torch.float16,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

#     # ------------------ Dataset ------------------ #
#     train_blocks = load_dataset_stream(
#         args.train_file,
#         tokenizer,
#         block_size=512,
#         max_blocks=450,  # 필요하면 늘리거나 None으로
#     )

#     # ------------------ DeepSpeed Engine Initialization ------------------ #
#     print("[DEBUG] Initializing DeepSpeed engine...")
#     ds_engine, optimizer, _, _ = deepspeed.initialize(
#         args=args,
#         model=model,
#         model_parameters=model.parameters(),
#     )

#     global_rank = ds_engine.global_rank
#     print(f"[DEBUG] DeepSpeed config (rank {global_rank}): {ds_engine.config}")

#     if ds_engine.checkpoint_engine:
#         print(f"[Rank {global_rank}] Using checkpoint engine: {type(ds_engine.checkpoint_engine).__name__}")
#     else:
#         print(f"[Rank {global_rank}] No checkpoint engine detected. Using default behavior.")

#     # ------------------ CSV Logger ------------------ #
#     log_f, log_writer = create_csv_logger(args.output_dir, global_rank)
#     if log_f is not None:
#         print(f"[DEBUG] Rank {global_rank} CSV log at {log_f.name}", flush=True)
#     else:
#         print(f"[DEBUG] Rank {global_rank} has no CSV logger (non-zero rank).", flush=True)

#     model.train()

#     max_steps = args.max_steps
#     ckpt_interval = args.checkpoint_interval

#     step = 0  # global step
#     for epoch in range(args.epochs):
#         print(f"[Rank {global_rank}] Entering epoch {epoch+1}")

#         for i, block in enumerate(train_blocks):
#             if step > max_steps:
#                 break

#             # ---- step 타이밍 시작 ----
#             step_start = time.time()

#             inputs = block.unsqueeze(0).to(ds_engine.device)
#             labels = inputs.clone()

#             # forward
#             outputs = ds_engine(inputs, labels=labels)
#             loss = outputs.loss

#             if global_rank == 0:
#                 print(f"[Rank 0] Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

#             # backward timing
#             bwd_start = time.time()
#             ds_engine.backward(loss)
#             bwd_end = time.time()
#             backward_seconds = bwd_end - bwd_start

#             # optimizer step
#             ds_engine.step()
#             step_end = time.time()
#             step_seconds = step_end - step_start

#             # ---- checkpoint 여부 결정 ----
#             # 🔴 여기서 200 step마다 + 마지막 step에서 체크포인트
#             checkpoint_now = (step % ckpt_interval == 0) or (step == max_steps)
#             # 만약 0,200,400,...,1000 만 원하면 위 or (step == max_steps) 를 지워도 됨

#             has_checkpoint = 0
#             total_ckpt_seconds = 0.0
#             enqueue_seconds = 0.0
#             flush_seconds = 0.0
#             barrier_seconds = 0.0
#             model_gpu2cpu_seconds = 0.0
#             model_cpu2disk_seconds = 0.0
#             optim_gpu2cpu_seconds = 0.0
#             optim_cpu2disk_seconds = 0.0
#             model_tensor_elements = 0
#             optim_tensor_elements = 0
#             model_logical_bytes = 0
#             optim_logical_bytes = 0
#             model_throughput_GBps = 0.0
#             optim_throughput_GBps = 0.0

#             if checkpoint_now:
#                 has_checkpoint = 1
#                 ckpt_dir = os.path.join(args.output_dir, f"checkpoint-1")

#                 tag = f"my_decoupled_ckpt_e{epoch+1}_s{step}"

#                 if global_rank == 0:
#                     print(f"[Rank {global_rank}] Starting save_checkpoint at step {step}...")

#                 t_ckpt_start = time.time()
#                 ds_engine.save_checkpoint(save_dir=ckpt_dir, tag=tag)
#                 # 필요하면 cleanup 호출
#                 if ds_engine.checkpoint_engine:
#                     ds_engine.checkpoint_engine.cleanup()
#                 t_ckpt_end = time.time()
#                 total_ckpt_seconds = t_ckpt_end - t_ckpt_start

#                 # 패치에서 기록해둔 timing/stat 가져오기
#                 stats = getattr(ds_engine.checkpoint_engine, "last_timing", {}) if ds_engine.checkpoint_engine else {}

#                 enqueue_seconds = float(stats.get("enqueue_seconds", 0.0))
#                 flush_seconds = float(stats.get("flush_seconds", 0.0))
#                 barrier_seconds = float(stats.get("barrier_seconds", 0.0))

#                 model_gpu2cpu_seconds = float(stats.get("model_gpu2cpu_seconds", 0.0))
#                 model_cpu2disk_seconds = float(stats.get("model_cpu2disk_seconds", 0.0))
#                 optim_gpu2cpu_seconds = float(stats.get("optim_gpu2cpu_seconds", 0.0))
#                 optim_cpu2disk_seconds = float(stats.get("optim_cpu2disk_seconds", 0.0))

#                 model_tensor_elements = int(stats.get("model_tensor_elements", 0))
#                 optim_tensor_elements = int(stats.get("optim_tensor_elements", 0))
#                 model_logical_bytes = int(stats.get("model_logical_bytes", 0))
#                 optim_logical_bytes = int(stats.get("optim_logical_bytes", 0))
#                 model_throughput_GBps = float(stats.get("model_throughput_GBps", 0.0))
#                 optim_throughput_GBps = float(stats.get("optim_throughput_GBps", 0.0))

#                 if global_rank == 0:
#                     print(f"[Rank {global_rank}] Total time to save and cleanup: {total_ckpt_seconds:.2f} seconds")

#             # ---- CSV 로그 한 줄 쓰기 ----
#             # 🔴 수정: 체크포인트가 있는 step 이고, rank 0에서만 로그
#             if log_writer is not None and has_checkpoint == 1 and global_rank == 0:
#                 ts = datetime.utcnow().isoformat()

#                 log_row = [
#                     ts,
#                     epoch + 1,
#                     step,
#                     float(loss.item()),
#                     global_rank,
#                     has_checkpoint,
#                     total_ckpt_seconds,
#                     enqueue_seconds,
#                     flush_seconds,
#                     barrier_seconds,
#                     model_gpu2cpu_seconds,
#                     model_cpu2disk_seconds,
#                     optim_gpu2cpu_seconds,
#                     optim_cpu2disk_seconds,
#                     model_tensor_elements,
#                     optim_tensor_elements,
#                     model_logical_bytes,
#                     optim_logical_bytes,
#                     model_throughput_GBps,
#                     optim_throughput_GBps,
#                     backward_seconds,
#                     step_seconds,
#                 ]

#                 log_writer.writerow(log_row)
#                 log_f.flush()
#                 print(f"[DEBUG] [Rank {global_rank}] Logged checkpoint row at step {step} to CSV.")

#             step += 1

#         if step > max_steps:
#             break

#     # 🔴 log_f 가 있을 때만 닫기 (rank 0)
#     if log_f is not None:
#         log_f.close()
#         print(f"[Rank {global_rank}] Training finished. CSV log closed.", flush=True)
#     else:
#         print(f"[Rank {global_rank}] Training finished. No CSV log for this rank.", flush=True)


# if __name__ == "__main__":
#     print("[DEBUG] >>> __main__ entered, calling main()")
#     main()



# import argparse
# import os
# import time
# import json
# import csv
# from datetime import datetime

# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# import torch
# import torch.distributed as dist
# import deepspeed
# from transformers import AutoTokenizer, AutoModelForCausalLM


# # ----------------------------------------------------
# #  Rank 0 전용 print
# # ----------------------------------------------------
# def r0_print(msg: str):
#     if os.environ.get("RANK", "0") == "0":
#         print(msg, flush=True)


# # ----------------------------------------------------
# #  텍스트 파일 → token block 리스트
# # ----------------------------------------------------
# def load_dataset_stream(file_path, tokenizer, block_size=512):
#     r0_print(f"[data] Loading dataset from: {file_path}")
#     blocks = []
#     buffer = []

#     with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#         for line in f:
#             if not line.strip():
#                 continue
#             tokens = tokenizer(
#                 line,
#                 return_tensors="pt",
#                 truncation=False
#             )["input_ids"].squeeze(0)
#             buffer.append(tokens)

#             # buffer에 토큰이 충분히 쌓이면 block_size 단위로 잘라서 블록 생성
#             while sum(b.numel() for b in buffer) >= block_size:
#                 concat = torch.cat(buffer)
#                 blocks.append(concat[:block_size])
#                 buffer = [concat[block_size:]] if concat.numel() > block_size else []

#     # 남은 토큰 처리 (필요시 padding)
#     if buffer:
#         concat = torch.cat(buffer)
#         pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
#         if concat.numel() < block_size:
#             concat = torch.nn.functional.pad(
#                 concat,
#                 (0, block_size - concat.numel()),
#                 value=pad_id
#             )
#         blocks.append(concat)

#     r0_print(f"[data] Total blocks created: {len(blocks)}")
#     return blocks


# # ----------------------------------------------------
# #  CSV 메트릭 (decoupled 스타일)
# # ----------------------------------------------------
# def init_csv_log(csv_path):
#     if not os.path.exists(os.path.dirname(csv_path)):
#         os.makedirs(os.path.dirname(csv_path), exist_ok=True)

#     if not os.path.exists(csv_path):
#         with open(csv_path, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 "timestamp",
#                 "step",
#                 "loss",
#                 "rank",

#                 "model_save_seconds",
#                 "optim_save_seconds",

#                 "model_disk_bytes",
#                 "optim_disk_bytes",

#                 "model_throughput_GBps",
#                 "optim_throughput_GBps",

#                 "backward_seconds",
#                 "step_seconds"
#             ])


# def append_csv_row(csv_path, row):
#     with open(csv_path, "a", newline="") as f:
#         csv.writer(f).writerow(row)


# # ----------------------------------------------------
# #  DeepSpeed 체크포인트 사이즈 계산 (rank0 기준)
# # ----------------------------------------------------
# def get_ckpt_file_sizes(ckpt_dir, tag, rank):
#     """
#     DeepSpeed save_checkpoint 구조를 가정하고,
#     rank0 기준으로 model / optimizer 파일 크기를 추정한다.
#     """
#     ckpt_path = os.path.join(ckpt_dir, tag)

#     # 모델 파일 (mp_rank_00_model_states.pt)
#     model_file = os.path.join(ckpt_path, "mp_rank_00_model_states.pt")
#     # ZeRO-2 optimizer 파일 (zero_pp_rank_0_mp_rank_00_optim_states.pt) 가정
#     optim_file = os.path.join(
#         ckpt_path,
#         f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
#     )

#     model_bytes = 0
#     optim_bytes = 0

#     if os.path.exists(model_file):
#         model_bytes = os.path.getsize(model_file)
#     else:
#         r0_print(f"[warn] Model file not found: {model_file}")

#     if os.path.exists(optim_file):
#         optim_bytes = os.path.getsize(optim_file)
#     else:
#         print(f"[warn][rank {rank}] Optim file not found: {optim_file}", flush=True)

#     return model_bytes, optim_bytes


# # ----------------------------------------------------
# #  MAIN
# # ----------------------------------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--deepspeed_config", type=str, required=True)
#     parser.add_argument("--model_name_or_path", type=str, default="bigscience/bloom-3b")
#     parser.add_argument("--train_file", type=str, required=True)
#     parser.add_argument("--output_dir", type=str, default="./bloom3b-deepspeed-decoupled")
#     parser.add_argument("--local_rank", type=int, default=-1)

#     # decoupled: step 기반
#     parser.add_argument("--total_steps", type=int, default=1070)
#     parser.add_argument("--save_interval", type=int, default=200)

#     # resume: DeepSpeed tag (예: step_400)
#     parser.add_argument("--resume_from", type=str, default=None)

#     args = parser.parse_args()

#     # -----------------------------
#     # rank / device 설정
#     # -----------------------------
#     args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
#     print(f"[boot] env RANK={os.environ.get('RANK')} local_rank={args.local_rank}", flush=True)
#     if args.local_rank >= 0:
#         torch.cuda.set_device(args.local_rank)

#     # -----------------------------
#     # DeepSpeed config 로드
#     # -----------------------------
#     with open(args.deepspeed_config, "r") as f:
#         ds_config = json.load(f)

#     r0_print("[config] Loaded DeepSpeed config:")
#     r0_print(json.dumps(ds_config, indent=2))

#     # -----------------------------
#     # 모델 & 토크나이저 로드
#     # -----------------------------
#     r0_print(f"[model] Loading model from {args.model_name_or_path}")
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_name_or_path,
#         device_map=None,
#         low_cpu_mem_usage=True,
#         torch_dtype=torch.float16,
#     )

#     # -----------------------------
#     # 데이터 블록 생성
#     # -----------------------------
#     base_blocks = load_dataset_stream(args.train_file, tokenizer, block_size=512)
#     if len(base_blocks) == 0:
#         raise RuntimeError("Dataset produced 0 blocks — training impossible.")

#     num_blocks = len(base_blocks)
#     total_steps = args.total_steps
#     save_interval = args.save_interval

#     r0_print(f"[data] num_blocks={num_blocks}, total_steps={total_steps}, save_interval={save_interval}")

#     # -----------------------------
#     # DeepSpeed 초기화
#     # -----------------------------
#     ds_engine, optimizer, _, _ = deepspeed.initialize(
#         model=model,
#         model_parameters=model.parameters(),
#         config=ds_config,
#     )

#     rank = ds_engine.global_rank
#     world_size = ds_engine.world_size
#     r0_print(f"[init] DeepSpeed initialized. world_size={world_size}")

#     # -----------------------------
#     # 출력 디렉토리 / CSV
#     # -----------------------------
#     ckpt_dir = os.path.join(args.output_dir, "checkpoints")
#     metrics_csv = os.path.join(ckpt_dir, "metrics_log.csv")

#     if rank == 0:
#         os.makedirs(ckpt_dir, exist_ok=True)
#         init_csv_log(metrics_csv)

#     # -----------------------------
#     # Resume-from (DeepSpeed load_checkpoint)
#     # -----------------------------
#     start_step = 0
#     if args.resume_from is not None:
#         r0_print(f"[resume] Trying to resume from tag={args.resume_from}")
#         # DeepSpeed의 tag = 디렉토리 이름 (예: step_400)
#         load_tag = args.resume_from

#         # DeepSpeed load_checkpoint 사용
#         ckpt_load_dir = ckpt_dir
#         load_path, client_state = ds_engine.load_checkpoint(
#             ckpt_load_dir,
#             tag=load_tag,
#             load_optimizer_states=True,
#             load_lr_scheduler_states=False,
#         )
#         r0_print(f"[resume] load_checkpoint path={load_path}, client_state={client_state}")

#         # client_state에 step 저장되어 있으면 그걸 사용
#         if client_state and "step" in client_state:
#             start_step = int(client_state["step"])
#         else:
#             # 태그가 step_400 이런 형식이면 그걸 파싱
#             if load_tag.startswith("step_"):
#                 start_step = int(load_tag.split("_")[1])

#         r0_print(f"[resume] Resuming from step={start_step}")

#     # -----------------------------
#     # TRAINING LOOP (step 기반)
#     # -----------------------------
#     r0_print(
#         f"[train] Start training: total_steps={total_steps}, "
#         f"save_interval={save_interval}, start_step={start_step}"
#     )

#     for step_idx in range(start_step, total_steps):
#         global_step = step_idx + 1

#         # 순환 방식으로 block 선택
#         block = base_blocks[step_idx % num_blocks]

#         # ---- Forward / Backward ----
#         t_bw_start = time.time()

#         inputs = block.unsqueeze(0).to(ds_engine.device)
#         labels = inputs.clone()

#         outputs = ds_engine(inputs, labels=labels)
#         loss = outputs.loss

#         ds_engine.backward(loss)
#         t_bw_end = time.time()

#         # ---- Optimizer step ----
#         t_step_start = time.time()
#         ds_engine.step()
#         t_step_end = time.time()

#         if (global_step % 50 == 0) and (rank == 0):
#             print(
#                 f"[train] Step {global_step}/{total_steps}, "
#                 f"Loss {loss.item():.4f}",
#                 flush=True
#             )

#         # -----------------------------
#         # Checkpoint 조건
#         # -----------------------------
#         is_last = (global_step == total_steps)
#         checkpoint_now = (global_step % save_interval == 0) or is_last

#         if checkpoint_now:
#             tag = f"step_{global_step}"

#             if dist.is_initialized():
#                 dist.barrier()

#             if rank == 0:
#                 print(f"[ckpt] >>> Saving DeepSpeed checkpoint: tag={tag} <<<", flush=True)

#             # DeepSpeed save_checkpoint 한 번 호출
#             t_ckpt_begin = time.time()

#             client_state = {"step": global_step}
#             save_path = ds_engine.save_checkpoint(
#                 ckpt_dir,
#                 tag=tag,
#                 client_state=client_state,
#                 save_latest=False
#             )

#             t_ckpt_end = time.time()
#             total_ckpt_seconds = t_ckpt_end - t_ckpt_begin

#             if dist.is_initialized():
#                 dist.barrier()

#             # rank0 기준으로 파일 크기 계산
#             model_disk_bytes = 0
#             optim_disk_bytes = 0
#             if rank == 0:
#                 model_disk_bytes, optim_disk_bytes = get_ckpt_file_sizes(ckpt_dir, tag, rank)
#                 print(
#                     f"[ckpt][rank0] tag={tag}, "
#                     f"time={total_ckpt_seconds:.3f}s, "
#                     f"model={model_disk_bytes/1e9:.3f} GB, "
#                     f"optim={optim_disk_bytes/1e9:.3f} GB, "
#                     f"path={save_path}",
#                     flush=True,
#                 )

#             # decoupled 스타일: model/optim 시간 분리할 수 없으므로
#             # total_ckpt_seconds를 둘 다에 기록
#             model_save_seconds = total_ckpt_seconds
#             optim_save_seconds = total_ckpt_seconds

#             # throughput 계산 (GB/s)
#             model_throughput = 0.0
#             if rank == 0 and model_save_seconds > 0 and model_disk_bytes > 0:
#                 model_throughput = (model_disk_bytes / model_save_seconds) / 1e9

#             optim_throughput = 0.0
#             if rank == 0 and optim_save_seconds > 0 and optim_disk_bytes > 0:
#                 optim_throughput = (optim_disk_bytes / optim_save_seconds) / 1e9

#             # CSV 기록 (rank0)
#             if rank == 0:
#                 append_csv_row(metrics_csv, [
#                     datetime.now().isoformat(),
#                     global_step,
#                     loss.item(),
#                     rank,

#                     model_save_seconds,
#                     optim_save_seconds,

#                     model_disk_bytes,
#                     optim_disk_bytes,

#                     model_throughput,
#                     optim_throughput,

#                     (t_bw_end - t_bw_start),
#                     (t_step_end - t_step_start),
#                 ])

#                 print(
#                     f"[ckpt][rank0] Logged metrics for {tag}: "
#                     f"model_throughput={model_throughput:.3f} GB/s, "
#                     f"optim_throughput={optim_throughput:.3f} GB/s",
#                     flush=True,
#                 )

#     r0_print("[train] Training finished.")


# if __name__ == "__main__":
#     main()


import argparse
import os
import time
import json
import csv
from datetime import datetime

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------------------------------------------
#  Rank 0 전용 print
# ----------------------------------------------------
def r0_print(msg: str):
    if os.environ.get("RANK", "0") == "0":
        print(msg, flush=True)


# ----------------------------------------------------
#  텍스트 파일 → token block 리스트
# ----------------------------------------------------
def load_dataset_stream(file_path, tokenizer, block_size=512):
    r0_print(f"[data] Loading dataset from: {file_path}")
    blocks = []
    buffer = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            tokens = tokenizer(
                line,
                return_tensors="pt",
                truncation=False
            )["input_ids"].squeeze(0)
            buffer.append(tokens)

            # buffer에 토큰이 충분히 쌓이면 block_size 단위로 잘라서 블록 생성
            while sum(b.numel() for b in buffer) >= block_size:
                concat = torch.cat(buffer)
                blocks.append(concat[:block_size])
                buffer = [concat[block_size:]] if concat.numel() > block_size else []

    # 남은 토큰 처리 (필요시 padding)
    if buffer:
        concat = torch.cat(buffer)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        if concat.numel() < block_size:
            concat = torch.nn.functional.pad(
                concat,
                (0, block_size - concat.numel()),
                value=pad_id
            )
        blocks.append(concat)

    r0_print(f"[data] Total blocks created: {len(blocks)}")
    return blocks


# ----------------------------------------------------
#  CSV 메트릭 (decoupled 스타일, 네가 준 포맷으로)
# ----------------------------------------------------
def init_csv_log(csv_path):
    csv_dir = os.path.dirname(csv_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "epoch",
                "step",
                "loss",
                "rank",

                "model_save_seconds",
                "optim_save_seconds",

                "model_tensor_elements",
                "optim_tensor_elements",

                "model_disk_bytes",
                "optim_disk_bytes",

                "model_throughput_GBps",
                "optim_throughput_GBps",

                "backward_seconds",
                "step_seconds",
            ])


def append_csv_row(csv_path, row):
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ----------------------------------------------------
#  DeepSpeed 체크포인트 사이즈 계산 (rank0 기준)
# ----------------------------------------------------
def get_ckpt_file_sizes(ckpt_dir, tag, rank):
    """
    DeepSpeed save_checkpoint 구조를 가정하고,
    rank0 기준으로 model / optimizer 파일 크기를 추정한다.
    """
    ckpt_path = os.path.join(ckpt_dir, tag)

    # 모델 파일 (mp_rank_00_model_states.pt)
    model_file = os.path.join(ckpt_path, "mp_rank_00_model_states.pt")
    # ZeRO-2 optimizer 파일 (zero_pp_rank_0_mp_rank_00_optim_states.pt) 가정
    optim_file = os.path.join(
        ckpt_path,
        f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
    )

    model_bytes = 0
    optim_bytes = 0

    if os.path.exists(model_file):
        model_bytes = os.path.getsize(model_file)
    else:
        r0_print(f"[warn] Model file not found: {model_file}")

    if os.path.exists(optim_file):
        optim_bytes = os.path.getsize(optim_file)
    else:
        print(f"[warn][rank {rank}] Optim file not found: {optim_file}", flush=True)

    return model_bytes, optim_bytes


# ----------------------------------------------------
#  MAIN
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed_config", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="bigscience/bloom-3b")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./bloom3b-deepspeed-decoupled")
    parser.add_argument("--local_rank", type=int, default=-1)

    # decoupled: step 기반
    parser.add_argument("--total_steps", type=int, default=1070)
    parser.add_argument("--save_interval", type=int, default=200)

    # resume: DeepSpeed tag (예: step_400)
    parser.add_argument("--resume_from", type=str, default=None)

    args = parser.parse_args()

    # -----------------------------
    # rank / device 설정
    # -----------------------------
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    print(f"[boot] env RANK={os.environ.get('RANK')} local_rank={args.local_rank}", flush=True)
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)

    # -----------------------------
    # DeepSpeed config 로드
    # -----------------------------
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    r0_print("[config] Loaded DeepSpeed config:")
    r0_print(json.dumps(ds_config, indent=2))

    # -----------------------------
    # 모델 & 토크나이저 로드
    # -----------------------------
    r0_print(f"[model] Loading model from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=None,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )

    # -----------------------------
    # 데이터 블록 생성
    # -----------------------------
    base_blocks = load_dataset_stream(args.train_file, tokenizer, block_size=512)
    if len(base_blocks) == 0:
        raise RuntimeError("Dataset produced 0 blocks — training impossible.")

    num_blocks = len(base_blocks)
    total_steps = args.total_steps
    save_interval = args.save_interval

    r0_print(f"[data] num_blocks={num_blocks}, total_steps={total_steps}, save_interval={save_interval}")

    # -----------------------------
    # DeepSpeed 초기화
    # -----------------------------
    ds_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    rank = ds_engine.global_rank
    world_size = ds_engine.world_size
    r0_print(f"[init] DeepSpeed initialized. world_size={world_size}")

    # -----------------------------
    # 출력 디렉토리 / CSV
    # -----------------------------
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    metrics_csv = os.path.join(ckpt_dir, "metrics_log.csv")

    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        init_csv_log(metrics_csv)

    # -----------------------------
    # Resume-from (DeepSpeed load_checkpoint)
    # -----------------------------
    start_step = 0
    if args.resume_from is not None:
        r0_print(f"[resume] Trying to resume from tag={args.resume_from}")
        # DeepSpeed의 tag = 디렉토리 이름 (예: step_400)
        load_tag = args.resume_from

        # DeepSpeed load_checkpoint 사용
        ckpt_load_dir = ckpt_dir
        load_path, client_state = ds_engine.load_checkpoint(
            ckpt_load_dir,
            tag=load_tag,
            load_optimizer_states=True,
            load_lr_scheduler_states=False,
        )
        r0_print(f"[resume] load_checkpoint path={load_path}, client_state={client_state}")

        # client_state에 step 저장되어 있으면 그걸 사용
        if client_state and "step" in client_state:
            start_step = int(client_state["step"])
        else:
            # 태그가 step_400 이런 형식이면 그걸 파싱
            if load_tag.startswith("step_"):
                start_step = int(load_tag.split("_")[1])

        r0_print(f"[resume] Resuming from step={start_step}")

    # -----------------------------
    # TRAINING LOOP (step 기반)
    # -----------------------------
    r0_print(
        f"[train] Start training: total_steps={total_steps}, "
        f"save_interval={save_interval}, start_step={start_step}"
    )

    fixed_epoch = 1  # 지금 로그처럼 epoch=1 고정

    for step_idx in range(start_step, total_steps):
        global_step = step_idx + 1

        # 순환 방식으로 block 선택
        block = base_blocks[step_idx % num_blocks]

        # ---- Forward / Backward ----
        t_bw_start = time.time()

        inputs = block.unsqueeze(0).to(ds_engine.device)
        labels = inputs.clone()

        outputs = ds_engine(inputs, labels=labels)
        loss = outputs.loss

        ds_engine.backward(loss)
        t_bw_end = time.time()

        # ---- Optimizer step ----
        t_step_start = time.time()
        ds_engine.step()
        t_step_end = time.time()

        if (global_step % 50 == 0) and (rank == 0):
            print(
                f"[train] Step {global_step}/{total_steps}, "
                f"Loss {loss.item():.4f}",
                flush=True
            )

        # -----------------------------
        # Checkpoint 조건
        # -----------------------------
        is_last = (global_step == total_steps)
        checkpoint_now = (global_step % save_interval == 0) or is_last

        if checkpoint_now:
            tag = f"step_{global_step}"

            if dist.is_initialized():
                dist.barrier()

            if rank == 0:
                print(f"[ckpt] >>> Saving DeepSpeed checkpoint: tag={tag} <<<", flush=True)

            # DeepSpeed save_checkpoint 한 번 호출
            t_ckpt_begin = time.time()

            client_state = {"step": global_step}
            save_path = ds_engine.save_checkpoint(
                ckpt_dir,
                tag=tag,
                client_state=client_state,
                save_latest=False
            )

            t_ckpt_end = time.time()
            total_ckpt_seconds = t_ckpt_end - t_ckpt_begin

            if dist.is_initialized():
                dist.barrier()

            # rank0 기준으로 파일 크기 계산
            model_disk_bytes = 0
            optim_disk_bytes = 0
            if rank == 0:
                model_disk_bytes, optim_disk_bytes = get_ckpt_file_sizes(ckpt_dir, tag, rank)
                print(
                    f"[ckpt][rank0] tag={tag}, "
                    f"time={total_ckpt_seconds:.3f}s, "
                    f"model={model_disk_bytes/1e9:.3f} GB, "
                    f"optim={optim_disk_bytes/1e9:.3f} GB, "
                    f"path={save_path}",
                    flush=True,
                )

            # decoupled: model/optim 시간을 분리할 수 없으니까
            # total_ckpt_seconds를 둘 다에 기록
            model_save_seconds = total_ckpt_seconds
            optim_save_seconds = total_ckpt_seconds

            # tensor elements (FP16 가정: bytes / 2)
            model_tensor_elements = 0
            optim_tensor_elements = 0
            if model_disk_bytes > 0:
                model_tensor_elements = model_disk_bytes // 2
            if optim_disk_bytes > 0:
                optim_tensor_elements = optim_disk_bytes // 2

            # throughput 계산 (GB/s)
            model_throughput = 0.0
            if model_save_seconds > 0 and model_disk_bytes > 0:
                model_throughput = (model_disk_bytes / model_save_seconds) / 1e9

            optim_throughput = 0.0
            if optim_save_seconds > 0 and optim_disk_bytes > 0:
                optim_throughput = (optim_disk_bytes / optim_save_seconds) / 1e9

            # CSV 기록 (rank0)
            if rank == 0:
                append_csv_row(metrics_csv, [
                    datetime.now().isoformat(),
                    fixed_epoch,
                    global_step,
                    loss.item(),
                    rank,

                    model_save_seconds,
                    optim_save_seconds,

                    model_tensor_elements,
                    optim_tensor_elements,

                    model_disk_bytes,
                    optim_disk_bytes,

                    model_throughput,
                    optim_throughput,

                    (t_bw_end - t_bw_start),
                    (t_step_end - t_step_start),
                ])

                print(
                    f"[ckpt][rank0] Logged metrics for {tag}: "
                    f"model_tensor_elements={model_tensor_elements}, "
                    f"optim_tensor_elements={optim_tensor_elements}, "
                    f"model_throughput={model_throughput:.6f} GB/s, "
                    f"optim_throughput={optim_throughput:.6f} GB/s",
                    flush=True,
                )

    r0_print("[train] Training finished.")


if __name__ == "__main__":
    main()