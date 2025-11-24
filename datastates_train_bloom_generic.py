# # import argparse
# # import os
# # import time
# # import json
# # import csv
# # from datetime import datetime

# # os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# # import torch
# # import deepspeed
# # from transformers import AutoTokenizer, AutoModelForCausalLM
# # from datastates.llm import Checkpointing


# # # ----------------------------------------------------
# # #  Move a nested state to CPU (with tensor stats)
# # # ----------------------------------------------------
# # def move_state_to_cpu(state, label="state"):
# #     tensor_count = 0
# #     total_numel = 0

# #     def _to_cpu(obj):
# #         nonlocal tensor_count, total_numel
# #         if isinstance(obj, torch.Tensor):
# #             tensor_count += 1
# #             total_numel += obj.numel()
# #             return obj.detach().cpu()
# #         elif isinstance(obj, dict):
# #             return {k: _to_cpu(v) for k, v in obj.items()}
# #         elif isinstance(obj, (list, tuple)):
# #             t = type(obj)
# #             return t(_to_cpu(v) for v in obj)
# #         else:
# #             return obj

# #     cpu_state = _to_cpu(state)
# #     print(
# #         f"[ckpt-debug] {label}: moved {tensor_count} tensors "
# #         f"({total_numel} elements) to CPU.",
# #         flush=True,
# #     )
# #     return cpu_state, tensor_count, total_numel


# # # ----------------------------------------------------
# # #  Tokenize text file → fixed-length blocks
# # # ----------------------------------------------------
# # def load_dataset_stream(file_path, tokenizer, block_size=512):
# #     blocks = []
# #     buffer = []
# #     with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
# #         for line in f:
# #             if not line.strip():
# #                 continue
# #             tokens = tokenizer(
# #                 line,
# #                 return_tensors="pt",
# #                 truncation=False
# #             )["input_ids"].squeeze(0)
# #             buffer.append(tokens)

# #             # 충분히 쌓이면 block_size 단위로 잘라서 블록으로 만듦
# #             while sum(b.numel() for b in buffer) >= block_size:
# #                 concat = torch.cat(buffer)
# #                 blocks.append(concat[:block_size])
# #                 buffer = [concat[block_size:]] if concat.numel() > block_size else []

# #     # 남은 토큰 처리 (필요하면 padding)
# #     if buffer:
# #         concat = torch.cat(buffer)
# #         pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
# #         if concat.numel() < block_size:
# #             concat = torch.nn.functional.pad(
# #                 concat,
# #                 (0, block_size - concat.numel()),
# #                 value=pad_id,
# #             )
# #         blocks.append(concat)

# #     return blocks


# # # ----------------------------------------------------
# # # Rank 0 only print
# # # ----------------------------------------------------
# # def r0_print(msg):
# #     if os.environ.get("RANK", "0") == "0":
# #         print(msg, flush=True)


# # # ----------------------------------------------------
# # # CSV metrics (gpu→cpu / cpu→disk / wait 분리)
# # # ----------------------------------------------------
# # def init_csv_log(csv_path):
# #     if not os.path.exists(csv_path):
# #         with open(csv_path, "w", newline="") as f:
# #             writer = csv.writer(f)
# #             writer.writerow([
# #                 "timestamp",
# #                 "epoch",
# #                 "step",
# #                 "loss",
# #                 "rank",

# #                 "total_ckpt_seconds",
# #                 "wait_seconds",              # ckpt_engine.wait() 동안 flush 대기 시간

# #                 "model_gpu2cpu_seconds",
# #                 "model_cpu2disk_seconds",
# #                 "optim_gpu2cpu_seconds",
# #                 "optim_cpu2disk_seconds",

# #                 "model_tensor_elements",
# #                 "optim_tensor_elements",

# #                 "model_logical_bytes",       # float16 기준 logical size
# #                 "optim_logical_bytes",       # float32 기준 logical size

# #                 "model_throughput_GBps",
# #                 "optim_throughput_GBps",

# #                 "backward_seconds",
# #                 "step_seconds",
# #             ])


# # def append_csv_row(csv_path, row):
# #     with open(csv_path, "a", newline="") as f:
# #         csv.writer(f).writerow(row)


# # # ----------------------------------------------------
# # # MAIN TRAINING LOOP
# # # ----------------------------------------------------
# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--deepspeed_config", type=str, required=True)
# #     parser.add_argument(
# #         "--model_name_or_path",
# #         type=str,
# #         default="bigscience/bloom-3b",
# #     )
# #     parser.add_argument("--train_file", type=str, required=True)
# #     parser.add_argument(
# #         "--output_dir",
# #         type=str,
# #         default="./bloom3b-finetuned",
# #     )
# #     parser.add_argument("--local_rank", type=int, default=-1)
# #     parser.add_argument("--epochs", type=int, default=1)
# #     parser.add_argument("--resume_from", type=str, default=None)
# #     parser.add_argument("--required_steps", type=int, default=1070)
# #     parser.add_argument("--block_size", type=int, default=512)
# #     args = parser.parse_args()

# #     # ------------------------------------------------
# #     # Rank & device 설정
# #     # ------------------------------------------------
# #     args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
# #     print(
# #         f"[boot] env RANK={os.environ.get('RANK')} "
# #         f"local_rank={args.local_rank}",
# #         flush=True,
# #     )
# #     if args.local_rank >= 0:
# #         torch.cuda.set_device(args.local_rank)

# #     # ------------------------------------------------
# #     # DeepSpeed config 로드
# #     # ------------------------------------------------
# #     with open(args.deepspeed_config, "r") as f:
# #         ds_config = json.load(f)

# #     ds_ckpt_cfg = ds_config.get("datastates_ckpt", {})
# #     ds_ckpt_cfg.setdefault("host_cache_size", 16)
# #     ds_ckpt_cfg.setdefault("parser_threads", 8)
# #     ds_config["datastates_ckpt"] = ds_ckpt_cfg

# #     r0_print("[Rank 0] datastates_ckpt:")
# #     r0_print(json.dumps(ds_ckpt_cfg, indent=2))

# #     # ------------------------------------------------
# #     # 모델 & 토크나이저 로드
# #     # ------------------------------------------------
# #     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
# #     model = AutoModelForCausalLM.from_pretrained(
# #         args.model_name_or_path,
# #         device_map=None,
# #         low_cpu_mem_usage=True,
# #         torch_dtype=torch.float16,
# #     )

# #     # ------------------------------------------------
# #     # 정확히 required_steps step 만들기
# #     # ------------------------------------------------
# #     base_blocks = load_dataset_stream(
# #         args.train_file,
# #         tokenizer,
# #         block_size=args.block_size,
# #     )
# #     if len(base_blocks) == 0:
# #         raise RuntimeError("Dataset produced 0 blocks — training impossible.")

# #     REQUIRED_STEPS = args.required_steps
# #     repeated = []
# #     while len(repeated) < REQUIRED_STEPS:
# #         repeated.extend(base_blocks)
# #     train_blocks = repeated[:REQUIRED_STEPS]

# #     r0_print(f"[Rank 0] Final block count = {len(train_blocks)}")

# #     # ------------------------------------------------
# #     # DeepSpeed 초기화
# #     # ------------------------------------------------
# #     ds_engine, optimizer, _, _ = deepspeed.initialize(
# #         model=model,
# #         model_parameters=model.parameters(),
# #         config=ds_config,
# #     )
# #     r0_print(f"[Rank {ds_engine.global_rank}] DeepSpeed initialized.")

# #     # ------------------------------------------------
# #     # DataStates 초기화
# #     # ------------------------------------------------
# #     ckpt_engine = Checkpointing(
# #         runtime_config=ds_ckpt_cfg,
# #         rank=ds_engine.global_rank,
# #     )

# #     # ------------------------------------------------
# #     # 출력 디렉토리 / CSV
# #     # ------------------------------------------------
# #     ckpt_dir = os.path.join(args.output_dir, "checkpoints")
# #     metrics_csv = os.path.join(ckpt_dir, "metrics_log.csv")

# #     if ds_engine.global_rank == 0:
# #         os.makedirs(ckpt_dir, exist_ok=True)
# #         init_csv_log(metrics_csv)

# #     # ------------------------------------------------
# #     # Resume-from (모델만 로드, optimizer는 새로 시작)
# #     # ------------------------------------------------
# #     start_epoch = 0
# #     start_step = 0

# #     if args.resume_from:
# #         tag_dir = os.path.join(ckpt_dir, args.resume_from)
# #         model_path = os.path.join(tag_dir, "mp_rank_00_model_states.pt")

# #         # model state 복원 (rank 0만)
# #         if ds_engine.global_rank == 0 and os.path.exists(model_path):
# #             r0_print(f"[Rank 0] Loading model from {model_path}")
# #             model_state = ckpt_engine.load(model_path)
# #             ds_engine.load_state_dict(model_state["model"])

# #         # 모든 rank 동기화
# #         torch.distributed.barrier()

# #         # optimizer.load_state_dict(...) 는 일부러 안 함
# #         # → ZeRO + CPU offload 구조와 충돌 방지

# #         if args.resume_from.startswith("step_"):
# #             start_step = int(args.resume_from.split("_")[1])
# #         elif args.resume_from.startswith("epoch_"):
# #             start_epoch = int(args.resume_from.split("_")[1]) - 1

# #         r0_print(
# #             f"[Rank {ds_engine.global_rank}] Resume from "
# #             f"epoch={start_epoch+1}, step={start_step}"
# #         )

# #     # ------------------------------------------------
# #     # TRAINING LOOP
# #     # ------------------------------------------------
# #     save_interval = 200
# #     r0_print(
# #         f"[Rank 0] Epochs={args.epochs}, save_interval={save_interval}, "
# #         f"steps/epoch={REQUIRED_STEPS}"
# #     )

# #     for epoch in range(start_epoch, args.epochs):
# #         r0_print(f"[Rank 0] Epoch {epoch+1} starting")

# #         for step, block in enumerate(train_blocks):

# #             if epoch == start_epoch and step < start_step:
# #                 continue

# #             # -------------------------
# #             # Forward / Backward
# #             # -------------------------
# #             t_bw_start = time.time()

# #             inputs = block.unsqueeze(0).to(ds_engine.device)
# #             labels = inputs.clone()

# #             outputs = ds_engine(inputs, labels=labels)
# #             loss = outputs.loss

# #             ds_engine.backward(loss)
# #             t_bw_end = time.time()

# #             # -------------------------
# #             # Optimizer step
# #             # -------------------------
# #             t_step_start = time.time()
# #             ds_engine.step()
# #             t_step_end = time.time()

# #             if step % 50 == 0 and ds_engine.global_rank == 0:
# #                 print(
# #                     f"[Rank 0] Epoch {epoch+1}, Step {step}, "
# #                     f"Loss {loss.item():.4f}",
# #                     flush=True,
# #                 )

# #             # -------------------------
# #             # Checkpoint 조건
# #             # -------------------------
# #             is_last = (step + 1 == REQUIRED_STEPS)
# #             checkpoint_now = (step % save_interval == 0) or is_last

# #             if checkpoint_now:
# #                 tag = f"epoch_{epoch+1}" if is_last else f"step_{step+1}"
# #                 tag_dir = os.path.join(ckpt_dir, tag)

# #                 if ds_engine.global_rank == 0:
# #                     os.makedirs(tag_dir, exist_ok=True)

# #                 # 모든 rank가 여기까지 온 뒤에 저장 시작
# #                 torch.distributed.barrier()

# #                 # === 0) 전체 CKPT 시간 타이머 시작 ===
# #                 t_save_begin = time.time()

# #                 # ------------------------------------------------
# #                 # 1) 모델 저장 (Rank 0만)
# #                 # ------------------------------------------------
# #                 model_path = None
# #                 model_logical_bytes = 0
# #                 model_gpu2cpu_seconds = 0.0
# #                 model_cpu2disk_seconds = 0.0
# #                 m_elements = 0

# #                 if ds_engine.global_rank == 0:
# #                     model_path = os.path.join(
# #                         tag_dir,
# #                         "mp_rank_00_model_states.pt",
# #                     )
# #                     print(
# #                         f"[Rank 0] Saving model → {model_path}",
# #                         flush=True,
# #                     )

# #                     raw_state = ds_engine.state_dict()

# #                     # GPU → CPU
# #                     t_m_gpu2cpu_begin = time.time()
# #                     m_cpu, m_tensors, m_elements = move_state_to_cpu(
# #                         raw_state,
# #                         "model",
# #                     )
# #                     t_m_gpu2cpu_end = time.time()
# #                     model_gpu2cpu_seconds = t_m_gpu2cpu_end - t_m_gpu2cpu_begin

# #                     # float16 기준 논리적 바이트 수
# #                     model_logical_bytes = m_elements * 2

# #                     # CPU → Disk (enqueue)
# #                     t_m_disk_begin = time.time()
# #                     ckpt_engine.save({"model": m_cpu}, model_path)
# #                     t_m_disk_end = time.time()
# #                     model_cpu2disk_seconds = t_m_disk_end - t_m_disk_begin

# #                     print(
# #                         f"[Rank 0][CKPT][model] "
# #                         f"gpu2cpu={model_gpu2cpu_seconds:.3f}s, "
# #                         f"cpu2disk(enqueue)={model_cpu2disk_seconds:.3f}s, "
# #                         f"logical_size≈{model_logical_bytes/1e9:.3f} GB",
# #                         flush=True,
# #                     )

# #                 # ------------------------------------------------
# #                 # 2) 옵티마이저 저장 (모든 rank)
# #                 # ------------------------------------------------
# #                 optim_path = os.path.join(
# #                     tag_dir,
# #                     f"zero_pp_rank_{ds_engine.global_rank}_"
# #                     f"mp_rank_00_optim_states.pt",
# #                 )
# #                 print(
# #                     f"[Rank {ds_engine.global_rank}] Saving optimizer → "
# #                     f"{optim_path}",
# #                     flush=True,
# #                 )

# #                 # GPU → CPU
# #                 t_o_gpu2cpu_begin = time.time()
# #                 raw_opt = optimizer.state_dict()
# #                 o_cpu, o_tensors, o_elements = move_state_to_cpu(
# #                     raw_opt,
# #                     f"optimizer_rank{ds_engine.global_rank}",
# #                 )
# #                 t_o_gpu2cpu_end = time.time()
# #                 optim_gpu2cpu_seconds = t_o_gpu2cpu_end - t_o_gpu2cpu_begin

# #                 # float32 기준 논리적 바이트 수
# #                 optim_logical_bytes = o_elements * 4

# #                 # CPU → Disk (enqueue)
# #                 t_o_disk_begin = time.time()
# #                 ckpt_engine.save({"optimizer": o_cpu}, optim_path)
# #                 t_o_disk_end = time.time()
# #                 optim_cpu2disk_seconds = t_o_disk_end - t_o_disk_begin

# #                 print(
# #                     f"[Rank {ds_engine.global_rank}][CKPT][optim] "
# #                     f"gpu2cpu={optim_gpu2cpu_seconds:.3f}s, "
# #                     f"cpu2disk(enqueue)={optim_cpu2disk_seconds:.3f}s, "
# #                     f"logical_size≈{optim_logical_bytes/1e9:.3f} GB",
# #                     flush=True,
# #                 )

# #                 # ------------------------------------------------
# #                 # 3) 모든 async I/O 완료 대기 (flush 구간)
# #                 # ------------------------------------------------
# #                 t_wait_begin = time.time()
# #                 ckpt_engine.wait()
# #                 t_wait_end = time.time()
# #                 wait_seconds = t_wait_end - t_wait_begin

# #                 print(
# #                     f"[Rank {ds_engine.global_rank}][CKPT] "
# #                     f"wait_for_io={wait_seconds:.3f}s",
# #                     flush=True,
# #                 )

# #                 # rank 간 barrier 한 번 더
# #                 torch.distributed.barrier()

# #                 # === 4) 전체 CKPT 시간 측정 끝 ===
# #                 t_save_end = time.time()
# #                 total_ckpt_seconds = t_save_end - t_save_begin

# #                 # ------------------------------------------------
# #                 # 5) Throughput 계산 (논리 바이트 기준)
# #                 # ------------------------------------------------
# #                 model_throughput = 0.0
# #                 if (
# #                     ds_engine.global_rank == 0
# #                     and total_ckpt_seconds > 0
# #                     and model_logical_bytes > 0
# #                 ):
# #                     model_throughput = (
# #                         model_logical_bytes / total_ckpt_seconds
# #                     ) / 1e9

# #                 optim_throughput = 0.0
# #                 if total_ckpt_seconds > 0 and optim_logical_bytes > 0:
# #                     optim_throughput = (
# #                         optim_logical_bytes / total_ckpt_seconds
# #                     ) / 1e9

# #                 # ------------------------------------------------
# #                 # 6) CSV 로깅 (Rank 0만) - optim은 rank0 shard 기준
# #                 # ------------------------------------------------
# #                 if ds_engine.global_rank == 0:
# #                     append_csv_row(metrics_csv, [
# #                         datetime.now().isoformat(),
# #                         epoch + 1,
# #                         step,
# #                         loss.item(),
# #                         ds_engine.global_rank,

# #                         total_ckpt_seconds,
# #                         wait_seconds,               # flush 대기 시간

# #                         model_gpu2cpu_seconds,
# #                         model_cpu2disk_seconds,
# #                         optim_gpu2cpu_seconds,
# #                         optim_cpu2disk_seconds,

# #                         m_elements,
# #                         o_elements,                 # rank0 optimizer shard only

# #                         model_logical_bytes,
# #                         optim_logical_bytes,

# #                         model_throughput,
# #                         optim_throughput,

# #                         (t_bw_end - t_bw_start),
# #                         (t_step_end - t_step_start),
# #                     ])

# #                     print(
# #                         f"[Rank 0] Checkpoint {tag} saved. "
# #                         f"model≈{model_logical_bytes/1e9:.3f} GB, "
# #                         f"optim_rank0≈{optim_logical_bytes/1e9:.3f} GB, "
# #                         f"total_ckpt={total_ckpt_seconds:.2f} s, "
# #                         f"wait={wait_seconds:.3f} s, "
# #                         f"model_throughput={model_throughput:.3f} GB/s, "
# #                         f"optim_throughput={optim_throughput:.3f} GB/s",
# #                         flush=True,
# #                     )

# #     r0_print("[Rank 0] Training finished.")


# # if __name__ == "__main__":
# #     main()

# import argparse
# import os
# import time
# import json
# import csv
# from datetime import datetime

# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# import torch
# import deepspeed
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from datastates.llm import Checkpointing


# # ----------------------------------------------------
# #  Move a nested state to CPU (with tensor stats)
# # ----------------------------------------------------
# def move_state_to_cpu(state, label="state"):
#     tensor_count = 0
#     total_numel = 0

#     def _to_cpu(obj):
#         nonlocal tensor_count, total_numel
#         if isinstance(obj, torch.Tensor):
#             tensor_count += 1
#             total_numel += obj.numel()
#             return obj.detach().cpu()
#         elif isinstance(obj, dict):
#             return {k: _to_cpu(v) for k, v in obj.items()}
#         elif isinstance(obj, (list, tuple)):
#             t = type(obj)
#             return t(_to_cpu(v) for v in obj)
#         else:
#             return obj

#     cpu_state = _to_cpu(state)
#     print(
#         f"[ckpt-debug] {label}: moved {tensor_count} tensors "
#         f"({total_numel} elements) to CPU.",
#         flush=True,
#     )
#     return cpu_state, tensor_count, total_numel


# # ----------------------------------------------------
# #  Tokenize text file → fixed-length blocks
# # ----------------------------------------------------
# def load_dataset_stream(file_path, tokenizer, block_size=512):
#     blocks = []
#     buffer = []
#     with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#         for line in f:
#             if not line.strip():
#                 continue
#             tokens = tokenizer(line, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
#             buffer.append(tokens)

#             # 충분히 쌓이면 block_size 단위로 잘라서 블록으로 만듦
#             while sum(b.numel() for b in buffer) >= block_size:
#                 concat = torch.cat(buffer)
#                 blocks.append(concat[:block_size])
#                 buffer = [concat[block_size:]] if concat.numel() > block_size else []

#     # 남은 토큰 처리 (필요하면 padding)
#     if buffer:
#         concat = torch.cat(buffer)
#         pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
#         if concat.numel() < block_size:
#             concat = torch.nn.functional.pad(concat, (0, block_size - concat.numel()), value=pad_id)
#         blocks.append(concat)

#     return blocks


# # ----------------------------------------------------
# # Rank 0 only print
# # ----------------------------------------------------
# def r0_print(msg):
#     if os.environ.get("RANK", "0") == "0":
#         print(msg, flush=True)


# # ----------------------------------------------------
# # CSV metrics (gpu→cpu / cpu→disk / enqueue / flush 분리)
# # ----------------------------------------------------
# def init_csv_log(csv_path):
#     if not os.path.exists(csv_path):
#         with open(csv_path, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 "timestamp",
#                 "epoch",
#                 "step",
#                 "loss",
#                 "rank",

#                 # 전체 CKPT 구간
#                 "total_ckpt_seconds",

#                 # enqueue 구간 vs flush+barrier 구간
#                 "enqueue_seconds",
#                 "flush_barrier_seconds",

#                 # 세부 breakdown
#                 "model_gpu2cpu_seconds",
#                 "model_cpu2disk_seconds",
#                 "optim_gpu2cpu_seconds",
#                 "optim_cpu2disk_seconds",

#                 "model_tensor_elements",
#                 "optim_tensor_elements",

#                 "model_logical_bytes",   # float16 기준 logical size
#                 "optim_logical_bytes",   # float32 기준 logical size

#                 "model_throughput_GBps",
#                 "optim_throughput_GBps",

#                 "backward_seconds",
#                 "step_seconds",
#             ])


# def append_csv_row(csv_path, row):
#     with open(csv_path, "a", newline="") as f:
#         csv.writer(f).writerow(row)


# # ----------------------------------------------------
# # MAIN TRAINING LOOP
# # ----------------------------------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--deepspeed_config", type=str, required=True)
#     parser.add_argument("--model_name_or_path", type=str,
#                         default="bigscience/bloom-3b")
#     parser.add_argument("--train_file", type=str, required=True)
#     parser.add_argument("--output_dir", type=str,
#                         default="./bloom3b-finetuned")
#     parser.add_argument("--local_rank", type=int, default=-1)
#     parser.add_argument("--epochs", type=int, default=1)
#     parser.add_argument("--resume_from", type=str, default=None)

#     # 실험 크기 조절용
#     parser.add_argument("--required_steps", type=int, default=1070)
#     parser.add_argument("--block_size", type=int, default=512)
#     args = parser.parse_args()

#     # ------------------------------------------------
#     # Rank & device 설정
#     # ------------------------------------------------
#     args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
#     print(f"[boot] env RANK={os.environ.get('RANK')} local_rank={args.local_rank}", flush=True)
#     if args.local_rank >= 0:
#         torch.cuda.set_device(args.local_rank)

#     # ------------------------------------------------
#     # DeepSpeed config 로드
#     # ------------------------------------------------
#     with open(args.deepspeed_config, "r") as f:
#         ds_config = json.load(f)

#     ds_ckpt_cfg = ds_config.get("datastates_ckpt", {})
#     ds_ckpt_cfg.setdefault("host_cache_size", 16)
#     ds_ckpt_cfg.setdefault("parser_threads", 8)
#     ds_config["datastates_ckpt"] = ds_ckpt_cfg

#     r0_print("[Rank 0] datastates_ckpt:")
#     r0_print(json.dumps(ds_ckpt_cfg, indent=2))

#     # ------------------------------------------------
#     # 모델 & 토크나이저 로드
#     # ------------------------------------------------
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_name_or_path,
#         device_map=None,
#         low_cpu_mem_usage=True,
#         torch_dtype=torch.float16,
#     )

#     # ------------------------------------------------
#     # 정확히 required_steps step 만들기
#     # ------------------------------------------------
#     base_blocks = load_dataset_stream(
#         args.train_file,
#         tokenizer,
#         block_size=args.block_size,
#     )
#     if len(base_blocks) == 0:
#         raise RuntimeError("Dataset produced 0 blocks — training impossible.")

#     REQUIRED_STEPS = args.required_steps
#     repeated = []
#     while len(repeated) < REQUIRED_STEPS:
#         repeated.extend(base_blocks)
#     train_blocks = repeated[:REQUIRED_STEPS]

#     r0_print(f"[Rank 0] Final block count = {len(train_blocks)}")

#     # ------------------------------------------------
#     # DeepSpeed 초기화
#     # ------------------------------------------------
#     ds_engine, optimizer, _, _ = deepspeed.initialize(
#         model=model,
#         model_parameters=model.parameters(),
#         config=ds_config,
#     )
#     r0_print(f"[Rank {ds_engine.global_rank}] DeepSpeed initialized.")

#     # ------------------------------------------------
#     # DataStates 초기화
#     # ------------------------------------------------
#     ckpt_engine = Checkpointing(runtime_config=ds_ckpt_cfg,
#                                 rank=ds_engine.global_rank)

#     # ------------------------------------------------
#     # 출력 디렉토리 / CSV
#     # ------------------------------------------------
#     ckpt_dir = os.path.join(args.output_dir, "checkpoints")
#     metrics_csv = os.path.join(ckpt_dir, "metrics_log.csv")

#     if ds_engine.global_rank == 0:
#         os.makedirs(ckpt_dir, exist_ok=True)
#         init_csv_log(metrics_csv)

#     # ------------------------------------------------
#     # Resume-from (모델만 로드, optimizer는 새로 시작)
#     # ------------------------------------------------
#     start_epoch = 0
#     start_step = 0

#     if args.resume_from:
#         tag_dir = os.path.join(ckpt_dir, args.resume_from)
#         model_path = os.path.join(tag_dir, "mp_rank_00_model_states.pt")

#         # model state 복원 (rank 0만)
#         if ds_engine.global_rank == 0 and os.path.exists(model_path):
#             r0_print(f"[Rank 0] Loading model from {model_path}")
#             model_state = ckpt_engine.load(model_path)
#             ds_engine.load_state_dict(model_state["model"])

#         # 모든 rank 동기화
#         torch.distributed.barrier()

#         # optimizer.load_state_dict(...) 는 일부러 안 함

#         if args.resume_from.startswith("step_"):
#             start_step = int(args.resume_from.split("_")[1])
#         elif args.resume_from.startswith("epoch_"):
#             start_epoch = int(args.resume_from.split("_")[1]) - 1

#         r0_print(
#             f"[Rank {ds_engine.global_rank}] Resume from "
#             f"epoch={start_epoch+1}, step={start_step}"
#         )

#     # ------------------------------------------------
#     # TRAINING LOOP
#     # ------------------------------------------------
#     save_interval = 200
#     r0_print(
#         f"[Rank 0] Epochs={args.epochs}, save_interval={save_interval}, "
#         f"steps/epoch={REQUIRED_STEPS}"
#     )

#     for epoch in range(start_epoch, args.epochs):
#         r0_print(f"[Rank 0] Epoch {epoch+1} starting")

#         for step, block in enumerate(train_blocks):

#             if epoch == start_epoch and step < start_step:
#                 continue

#             # -------------------------
#             # Forward / Backward
#             # -------------------------
#             t_bw_start = time.time()

#             inputs = block.unsqueeze(0).to(ds_engine.device)
#             labels = inputs.clone()

#             outputs = ds_engine(inputs, labels=labels)
#             loss = outputs.loss

#             ds_engine.backward(loss)
#             t_bw_end = time.time()

#             # -------------------------
#             # Optimizer step
#             # -------------------------
#             t_step_start = time.time()
#             ds_engine.step()
#             t_step_end = time.time()

#             if step % 50 == 0 and ds_engine.global_rank == 0:
#                 print(
#                     f"[Rank 0] Epoch {epoch+1}, Step {step}, "
#                     f"Loss {loss.item():.4f}",
#                     flush=True,
#                 )

#             # -------------------------
#             # Checkpoint 조건
#             # -------------------------
#             is_last = (step + 1 == REQUIRED_STEPS)
#             checkpoint_now = (step % save_interval == 0) or is_last

#             if checkpoint_now:
#                 tag = f"epoch_{epoch+1}" if is_last else f"step_{step+1}"
#                 tag_dir = os.path.join(ckpt_dir, tag)

#                 if ds_engine.global_rank == 0:
#                     os.makedirs(tag_dir, exist_ok=True)

#                 # 모든 rank가 여기까지 온 뒤에 저장 시작
#                 torch.distributed.barrier()

#                 # === 0) 전체 CKPT 시간 타이머 시작 ===
#                 t_save_begin = time.time()

#                 # ------------------------------------------------
#                 # 1) 모델 저장 (Rank 0만)
#                 # ------------------------------------------------
#                 model_path = None
#                 model_logical_bytes = 0
#                 model_gpu2cpu_seconds = 0.0
#                 model_cpu2disk_seconds = 0.0
#                 m_elements = 0

#                 if ds_engine.global_rank == 0:
#                     model_path = os.path.join(
#                         tag_dir,
#                         "mp_rank_00_model_states.pt",
#                     )
#                     print(f"[Rank 0] Saving model → {model_path}", flush=True)

#                     raw_state = ds_engine.state_dict()

#                     # GPU → CPU
#                     t_m_gpu2cpu_begin = time.time()
#                     m_cpu, m_tensors, m_elements = move_state_to_cpu(
#                         raw_state, "model"
#                     )
#                     t_m_gpu2cpu_end = time.time()
#                     model_gpu2cpu_seconds = t_m_gpu2cpu_end - t_m_gpu2cpu_begin

#                     # float16 기준 논리적 바이트 수
#                     model_logical_bytes = m_elements * 2

#                     # CPU → Disk (enqueue)
#                     t_m_disk_begin = time.time()
#                     ckpt_engine.save({"model": m_cpu}, model_path)
#                     t_m_disk_end = time.time()
#                     model_cpu2disk_seconds = t_m_disk_end - t_m_disk_begin

#                     print(
#                         f"[Rank 0][CKPT][model] "
#                         f"gpu2cpu={model_gpu2cpu_seconds:.3f}s, "
#                         f"cpu2disk(enqueue)={model_cpu2disk_seconds:.3f}s, "
#                         f"logical_size≈{model_logical_bytes/1e9:.3f} GB",
#                         flush=True,
#                     )

#                 # ------------------------------------------------
#                 # 2) 옵티마이저 저장 (모든 rank)
#                 # ------------------------------------------------
#                 optim_path = os.path.join(
#                     tag_dir,
#                     f"zero_pp_rank_{ds_engine.global_rank}_"
#                     f"mp_rank_00_optim_states.pt",
#                 )
#                 print(
#                     f"[Rank {ds_engine.global_rank}] Saving optimizer → "
#                     f"{optim_path}",
#                     flush=True,
#                 )

#                 # GPU → CPU
#                 t_o_gpu2cpu_begin = time.time()
#                 raw_opt = optimizer.state_dict()
#                 o_cpu, o_tensors, o_elements = move_state_to_cpu(
#                     raw_opt,
#                     f"optimizer_rank{ds_engine.global_rank}",
#                 )
#                 t_o_gpu2cpu_end = time.time()
#                 optim_gpu2cpu_seconds = t_o_gpu2cpu_end - t_o_gpu2cpu_begin

#                 # float32 기준 논리적 바이트 수
#                 optim_logical_bytes = o_elements * 4

#                 # CPU → Disk (enqueue)
#                 t_o_disk_begin = time.time()
#                 ckpt_engine.save({"optimizer": o_cpu}, optim_path)
#                 t_o_disk_end = time.time()
#                 optim_cpu2disk_seconds = t_o_disk_end - t_o_disk_begin

#                 print(
#                     f"[Rank {ds_engine.global_rank}][CKPT][optim] "
#                         f"gpu2cpu={optim_gpu2cpu_seconds:.3f}s, "
#                         f"cpu2disk(enqueue)={optim_cpu2disk_seconds:.3f}s, "
#                         f"logical_size≈{optim_logical_bytes/1e9:.3f} GB",
#                     flush=True,
#                 )

#                 # === enqueue phase 끝난 시점 ===
#                 t_enqueue_end = time.time()

#                 # ------------------------------------------------
#                 # 3) 모든 async I/O 완료 대기 (flush)
#                 # ------------------------------------------------
#                 ckpt_engine.wait()

#                 # rank 간 barrier 한 번 더
#                 torch.distributed.barrier()

#                 # === 4) 전체 CKPT 시간 측정 끝 ===
#                 t_save_end = time.time()

#                 total_ckpt_seconds = t_save_end - t_save_begin
#                 enqueue_seconds = t_enqueue_end - t_save_begin
#                 flush_barrier_seconds = total_ckpt_seconds - enqueue_seconds

#                 # ------------------------------------------------
#                 # 5) Throughput 계산 (논리 바이트 기준)
#                 # ------------------------------------------------
#                 model_throughput = 0.0
#                 if (ds_engine.global_rank == 0 and
#                         total_ckpt_seconds > 0 and
#                         model_logical_bytes > 0):
#                     model_throughput = (
#                         model_logical_bytes / total_ckpt_seconds
#                     ) / 1e9

#                 optim_throughput = 0.0
#                 if total_ckpt_seconds > 0 and optim_logical_bytes > 0:
#                     optim_throughput = (
#                         optim_logical_bytes / total_ckpt_seconds
#                     ) / 1e9

#                 # ------------------------------------------------
#                 # 6) CSV 로깅 (Rank 0만) - optim은 rank0 shard 기준
#                 # ------------------------------------------------
#                 if ds_engine.global_rank == 0:
#                     append_csv_row(metrics_csv, [
#                         datetime.now().isoformat(),
#                         epoch + 1,
#                         step,
#                         loss.item(),
#                         ds_engine.global_rank,

#                         total_ckpt_seconds,
#                         enqueue_seconds,
#                         flush_barrier_seconds,

#                         model_gpu2cpu_seconds,
#                         model_cpu2disk_seconds,
#                         optim_gpu2cpu_seconds,
#                         optim_cpu2disk_seconds,

#                         m_elements,
#                         o_elements,          # rank0 optimizer shard only

#                         model_logical_bytes,
#                         optim_logical_bytes,

#                         model_throughput,
#                         optim_throughput,

#                         (t_bw_end - t_bw_start),
#                         (t_step_end - t_step_start),
#                     ])

#                     print(
#                         f"[Rank 0] Checkpoint {tag} saved. "
#                         f"model≈{model_logical_bytes/1e9:.3f} GB, "
#                         f"optim_rank0≈{optim_logical_bytes/1e9:.3f} GB, "
#                         f"total_ckpt={total_ckpt_seconds:.3f} s, "
#                         f"enqueue={enqueue_seconds:.3f} s, "
#                         f"flush+barrier={flush_barrier_seconds:.3f} s, "
#                         f"model_throughput={model_throughput:.3f} GB/s, "
#                         f"optim_throughput={optim_throughput:.3f} GB/s",
#                         flush=True,
#                     )

#     r0_print("[Rank 0] Training finished.")


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
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
from datastates.llm import Checkpointing


# ----------------------------------------------------
#  Move a nested state to CPU (with tensor stats)
# ----------------------------------------------------
def move_state_to_cpu(state, label="state"):
    tensor_count = 0
    total_numel = 0

    def _to_cpu(obj):
        nonlocal tensor_count, total_numel
        if isinstance(obj, torch.Tensor):
            tensor_count += 1
            total_numel += obj.numel()
            return obj.detach().cpu()
        elif isinstance(obj, dict):
            return {k: _to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            t = type(obj)
            return t(_to_cpu(v) for v in obj)
        else:
            return obj

    cpu_state = _to_cpu(state)
    print(
        f"[ckpt-debug] {label}: moved {tensor_count} tensors "
        f"({total_numel} elements) to CPU.",
        flush=True,
    )
    return cpu_state, tensor_count, total_numel


# ----------------------------------------------------
#  Tokenize text file → fixed-length blocks
# ----------------------------------------------------
def load_dataset_stream(file_path, tokenizer, block_size=512):
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

            # 충분히 쌓이면 block_size 단위로 잘라서 블록으로 만듦
            while sum(b.numel() for b in buffer) >= block_size:
                concat = torch.cat(buffer)
                blocks.append(concat[:block_size])
                buffer = [concat[block_size:]] if concat.numel() > block_size else []

    # 남은 토큰 처리 (필요하면 padding)
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


# ----------------------------------------------------
# Rank 0 only print
# ----------------------------------------------------
def r0_print(msg: str):
    if os.environ.get("RANK", "0") == "0":
        print(msg, flush=True)


# ----------------------------------------------------
# CSV metrics (gpu→cpu / cpu→disk / enqueue / flush 구분)
# ----------------------------------------------------
def init_csv_log(csv_path: str):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "epoch",
                "step",
                "loss",
                "rank",

                "total_ckpt_seconds",
                "enqueue_seconds",
                "flush_barrier_seconds",

                "model_gpu2cpu_seconds",
                "model_cpu2disk_seconds",
                "optim_gpu2cpu_seconds",
                "optim_cpu2disk_seconds",

                "model_tensor_elements",
                "optim_tensor_elements",

                "model_logical_bytes",   # float16 기준 logical size
                "optim_logical_bytes",   # float32 기준 logical size

                "model_throughput_GBps",
                "optim_throughput_GBps",

                "backward_seconds",
                "step_seconds",
            ])


def append_csv_row(csv_path: str, row):
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ----------------------------------------------------
# MAIN TRAINING LOOP
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed_config", type=str, required=True)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bigscience/bloom-3b",
    )
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./bloom3b-finetuned",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--required_steps", type=int, default=1070)
    parser.add_argument("--block_size", type=int, default=512)
    args = parser.parse_args()

    # ------------------------------------------------
    # Rank & device 설정
    # ------------------------------------------------
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    print(
        f"[boot] env RANK={os.environ.get('RANK')} "
        f"local_rank={args.local_rank}",
        flush=True,
    )
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)

    # ------------------------------------------------
    # DeepSpeed config 로드
    # ------------------------------------------------
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_ckpt_cfg = ds_config.get("datastates_ckpt", {})
    ds_ckpt_cfg.setdefault("host_cache_size", 16)   # GB
    ds_ckpt_cfg.setdefault("parser_threads", 8)
    ds_config["datastates_ckpt"] = ds_ckpt_cfg

    r0_print("[Rank 0] datastates_ckpt:")
    r0_print(json.dumps(ds_ckpt_cfg, indent=2))

    # ------------------------------------------------
    # 모델 & 토크나이저 로드
    # ------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=None,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )

    # ------------------------------------------------
    # 정확히 required_steps step 만들기
    # ------------------------------------------------
    base_blocks = load_dataset_stream(
        args.train_file,
        tokenizer,
        block_size=args.block_size,
    )
    if len(base_blocks) == 0:
        raise RuntimeError("Dataset produced 0 blocks — training impossible.")

    REQUIRED_STEPS = args.required_steps
    repeated = []
    while len(repeated) < REQUIRED_STEPS:
        repeated.extend(base_blocks)
    train_blocks = repeated[:REQUIRED_STEPS]

    r0_print(f"[Rank 0] Final block count = {len(train_blocks)}")

    # ------------------------------------------------
    # DeepSpeed 초기화
    # ------------------------------------------------
    ds_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )
    r0_print(f"[Rank {ds_engine.global_rank}] DeepSpeed initialized.")

    # ------------------------------------------------
    # DataStates 초기화
    # ------------------------------------------------
    ckpt_engine = Checkpointing(
        runtime_config=ds_ckpt_cfg,
        rank=ds_engine.global_rank,
    )

    # ------------------------------------------------
    # 출력 디렉토리 / CSV
    # ------------------------------------------------
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    metrics_csv = os.path.join(ckpt_dir, "metrics_log.csv")

    if ds_engine.global_rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        init_csv_log(metrics_csv)

    # ------------------------------------------------
    # Resume-from (모델만 로드, optimizer는 새로 시작)
    # ------------------------------------------------
    start_epoch = 0
    start_step = 0

    if args.resume_from:
        tag_dir = os.path.join(ckpt_dir, args.resume_from)
        model_path = os.path.join(tag_dir, "mp_rank_00_model_states.pt")

        # model state 복원 (rank 0만)
        if ds_engine.global_rank == 0 and os.path.exists(model_path):
            r0_print(f"[Rank 0] Loading model from {model_path}")
            model_state = ckpt_engine.load(model_path)
            ds_engine.load_state_dict(model_state["model"])

        # 모든 rank 동기화
        torch.distributed.barrier()

        # optimizer.load_state_dict(...) 는 일부러 안 함
        # → ZeRO+CPU offload 구조와 충돌 방지용

        if args.resume_from.startswith("step_"):
            start_step = int(args.resume_from.split("_")[1])
        elif args.resume_from.startswith("epoch_"):
            start_epoch = int(args.resume_from.split("_")[1]) - 1

        r0_print(
            f"[Rank {ds_engine.global_rank}] Resume from "
            f"epoch={start_epoch+1}, step={start_step}"
        )

    # ------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------
    save_interval = 200
    r0_print(
        f"[Rank 0] Epochs={args.epochs}, save_interval={save_interval}, "
        f"steps/epoch={REQUIRED_STEPS}"
    )

    for epoch in range(start_epoch, args.epochs):
        r0_print(f"[Rank 0] Epoch {epoch+1} starting")

        for step, block in enumerate(train_blocks):

            if epoch == start_epoch and step < start_step:
                continue

            # -------------------------
            # Forward / Backward
            # -------------------------
            t_bw_start = time.time()

            inputs = block.unsqueeze(0).to(ds_engine.device)
            labels = inputs.clone()

            outputs = ds_engine(inputs, labels=labels)
            loss = outputs.loss

            ds_engine.backward(loss)
            t_bw_end = time.time()

            # -------------------------
            # Optimizer step
            # -------------------------
            t_step_start = time.time()
            ds_engine.step()
            t_step_end = time.time()

            if step % 50 == 0 and ds_engine.global_rank == 0:
                print(
                    f"[Rank 0] Epoch {epoch+1}, Step {step}, "
                    f"Loss {loss.item():.4f}",
                    flush=True,
                )

            # -------------------------
            # Checkpoint 조건
            # -------------------------
            is_last = (step + 1 == REQUIRED_STEPS)
            checkpoint_now = (step % save_interval == 0) or is_last

            if checkpoint_now:
                tag = f"epoch_{epoch+1}" if is_last else f"step_{step+1}"
                tag_dir = os.path.join(ckpt_dir, tag)

                if ds_engine.global_rank == 0:
                    os.makedirs(tag_dir, exist_ok=True)

                # 모든 rank가 여기까지 온 뒤에 저장 시작
                torch.distributed.barrier()

                # === 0) 전체 CKPT 시간 타이머 시작 ===
                t_save_begin = time.time()

                # ------------------------------------------------
                # 1) 모델 저장 (Rank 0만)
                # ------------------------------------------------
                model_path = None
                model_logical_bytes = 0
                model_gpu2cpu_seconds = 0.0
                model_cpu2disk_seconds = 0.0
                m_elements = 0

                if ds_engine.global_rank == 0:
                    model_path = os.path.join(
                        tag_dir,
                        "mp_rank_00_model_states.pt",
                    )
                    print(f"[Rank 0] Saving model → {model_path}", flush=True)

                    raw_state = ds_engine.state_dict()

                    # GPU → CPU
                    t_m_gpu2cpu_begin = time.time()
                    m_cpu, m_tensors, m_elements = move_state_to_cpu(
                        raw_state,
                        "model",
                    )
                    t_m_gpu2cpu_end = time.time()
                    model_gpu2cpu_seconds = t_m_gpu2cpu_end - t_m_gpu2cpu_begin

                    # float16 기준 논리적 바이트 수
                    model_logical_bytes = m_elements * 2

                    # CPU → Disk (enqueue)
                    t_m_disk_begin = time.time()
                    ckpt_engine.save({"model": m_cpu}, model_path)
                    t_m_disk_end = time.time()
                    model_cpu2disk_seconds = t_m_disk_end - t_m_disk_begin

                    print(
                        f"[Rank 0][CKPT][model] "
                        f"gpu2cpu={model_gpu2cpu_seconds:.3f}s, "
                        f"cpu2disk(enqueue)={model_cpu2disk_seconds:.3f}s, "
                        f"logical_size≈{model_logical_bytes/1e9:.3f} GB",
                        flush=True,
                    )

                # ------------------------------------------------
                # 2) 옵티마이저 저장 (모든 rank)
                # ------------------------------------------------
                optim_path = os.path.join(
                    tag_dir,
                    f"zero_pp_rank_{ds_engine.global_rank}_"
                    f"mp_rank_00_optim_states.pt",
                )
                print(
                    f"[Rank {ds_engine.global_rank}] Saving optimizer → "
                    f"{optim_path}",
                    flush=True,
                )

                # GPU → CPU
                t_o_gpu2cpu_begin = time.time()
                raw_opt = optimizer.state_dict()
                o_cpu, o_tensors, o_elements = move_state_to_cpu(
                    raw_opt,
                    f"optimizer_rank{ds_engine.global_rank}",
                )
                t_o_gpu2cpu_end = time.time()
                optim_gpu2cpu_seconds = t_o_gpu2cpu_end - t_o_gpu2cpu_begin

                # float32 기준 논리적 바이트 수
                optim_logical_bytes = o_elements * 4

                # CPU → Disk (enqueue)
                t_o_disk_begin = time.time()
                ckpt_engine.save({"optimizer": o_cpu}, optim_path)
                t_o_disk_end = time.time()
                optim_cpu2disk_seconds = t_o_disk_end - t_o_disk_begin

                print(
                    f"[Rank {ds_engine.global_rank}][CKPT][optim] "
                    f"gpu2cpu={optim_gpu2cpu_seconds:.3f}s, "
                    f"cpu2disk(enqueue)={optim_cpu2disk_seconds:.3f}s, "
                    f"logical_size≈{optim_logical_bytes/1e9:.3f} GB",
                    flush=True,
                )

                # ------------------------------------------------
                # 3) enqueue 구간 끝 / flush+barrier 구간 시작
                # ------------------------------------------------
                t_after_enqueue = time.time()

                # 모든 async I/O 완료 대기 + rank barrier
                ckpt_engine.wait()
                torch.distributed.barrier()

                # === 4) 전체 CKPT 시간 측정 끝 ===
                t_save_end = time.time()
                total_ckpt_seconds = t_save_end - t_save_begin
                enqueue_seconds = t_after_enqueue - t_save_begin
                flush_barrier_seconds = total_ckpt_seconds - enqueue_seconds

                # ------------------------------------------------
                # 5) Throughput 계산 (논리 바이트 기준)
                # ------------------------------------------------
                model_throughput = 0.0
                if (
                    ds_engine.global_rank == 0
                    and total_ckpt_seconds > 0
                    and model_logical_bytes > 0
                ):
                    model_throughput = (
                        model_logical_bytes / total_ckpt_seconds
                    ) / 1e9

                optim_throughput = 0.0
                if total_ckpt_seconds > 0 and optim_logical_bytes > 0:
                    optim_throughput = (
                        optim_logical_bytes / total_ckpt_seconds
                    ) / 1e9

                # ------------------------------------------------
                # 6) CSV 로깅 (Rank 0만)
                # ------------------------------------------------
                if ds_engine.global_rank == 0:
                    append_csv_row(metrics_csv, [
                        datetime.now().isoformat(),
                        epoch + 1,
                        step,
                        loss.item(),
                        ds_engine.global_rank,

                        total_ckpt_seconds,
                        enqueue_seconds,
                        flush_barrier_seconds,

                        model_gpu2cpu_seconds,
                        model_cpu2disk_seconds,
                        optim_gpu2cpu_seconds,
                        optim_cpu2disk_seconds,

                        m_elements,
                        o_elements,          # rank0 optimizer shard only

                        model_logical_bytes,
                        optim_logical_bytes,

                        model_throughput,
                        optim_throughput,

                        (t_bw_end - t_bw_start),
                        (t_step_end - t_step_start),
                    ])

                    print(
                        f"[Rank 0] Checkpoint {tag} saved. "
                        f"model≈{model_logical_bytes/1e9:.3f} GB, "
                        f"optim_rank0≈{optim_logical_bytes/1e9:.3f} GB, "
                        f"total_ckpt={total_ckpt_seconds:.2f} s, "
                        f"model_throughput={model_throughput:.3f} GB/s, "
                        f"optim_throughput={optim_throughput:.3f} GB/s",
                        flush=True,
                    )

    r0_print("[Rank 0] Training finished.")


if __name__ == "__main__":
    main()