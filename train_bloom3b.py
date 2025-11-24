# # # import os
# # # import time
# # # import torch
# # # import argparse
# # # import deepspeed
# # # from transformers import AutoModelForCausalLM, AutoTokenizer

# # # # Parse arguments for DeepSpeed integration
# # # parser = argparse.ArgumentParser()
# # # parser.add_argument('--deepspeed_config', type=str, required=True, help='Path to DeepSpeed config file')
# # # parser.add_argument('--model_name_or_path', type=str, default='bigscience/bloom-3b', help='HuggingFace model name or path')
# # # parser.add_argument('--train_file', type=str, required=True, help='Path to a plain text training file')
# # # parser.add_argument('--output_dir', type=str, default='./bloom3b-finetuned')
# # # parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed by deepspeed/torch.distributed')
# # # parser.add_argument('--per_device_train_batch_size', type=int, default=2, help='Batch size per device/GPU')
# # # parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
# # # parser.add_argument('--master_port', type=int, default=None, help='Master port for distributed training')
# # # parser.add_argument('--resume_from', type=str, default=None, help="Resume training from this checkpoint tag (e.g., 'epoch_3'). If not set, start from scratch.")
# # # args = parser.parse_args()

# # # # Prefer LOCAL_RANK from environment (set by srun/DeepSpeed) when available.
# # # args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))

# # # if args.local_rank is not None and args.local_rank >= 0:
# # #     torch.cuda.set_device(args.local_rank)

# # # model = AutoModelForCausalLM.from_pretrained(
# # #     args.model_name_or_path,
# # #     device_map=None,
# # #     low_cpu_mem_usage=True,
# # #     torch_dtype=torch.float16
# # # )
# # # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

# # # def load_dataset_stream(file_path, tokenizer, block_size=512, max_blocks=None):
# # #     blocks = []
# # #     buffer = []
# # #     with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
# # #         for line in f:
# # #             if not line.strip():
# # #                 continue
# # #             tokens = tokenizer(line, return_tensors='pt', truncation=False)["input_ids"].squeeze(0)
# # #             buffer.append(tokens)
# # #             while sum(b.numel() for b in buffer) >= block_size:
# # #                 concat = torch.cat(buffer)
# # #                 blocks.append(concat[:block_size])
# # #                 buffer = [concat[block_size:]] if concat.numel() > block_size else []
# # #                 if max_blocks is not None and len(blocks) >= max_blocks:
# # #                     return blocks
# # #     if buffer and (max_blocks is None or len(blocks) < max_blocks):
# # #         concat = torch.cat(buffer)
# # #         if concat.numel() >= 1:
# # #             concat = torch.nn.functional.pad(concat, (0, block_size - concat.numel()), value=tokenizer.pad_token_id)
# # #             blocks.append(concat)
# # #     return blocks

# # # train_blocks = load_dataset_stream(args.train_file, tokenizer, block_size=512, max_blocks=450)

# # # ds_engine, optimizer, _, _ = deepspeed.initialize(
# # #     model=model,
# # #     model_parameters=model.parameters(),
# # #     config=args.deepspeed_config
# # # )

# # # model.train()

# # # save_interval = 500
# # # ckpt_dir = os.path.join(args.output_dir, "checkpoints")
# # # if ds_engine.global_rank == 0:
# # #     os.makedirs(ckpt_dir, exist_ok=True)

# # # # --- Resume logic ---
# # # latest_epoch = 0
# # # if args.resume_from is not None:
# # #     print(f"[Rank {ds_engine.global_rank}] Attempting to resume from checkpoint: {args.resume_from}", flush=True)
# # #     load_start = time.time()
# # #     load_success, client_state = ds_engine.load_checkpoint(ckpt_dir, tag=args.resume_from)
# # #     load_time = time.time() - load_start
# # #     if ds_engine.global_rank == 0:
# # #         print(f"[Rank 0] Checkpoint loaded in {load_time:.2f}s", flush=True)
# # #     if load_success:
# # #         latest_epoch = int(args.resume_from.split("_")[1])
# # #         print(f"[Rank {ds_engine.global_rank}] Successfully resumed from {args.resume_from}, continuing at epoch {latest_epoch+1}")
# # #     else:
# # #         print(f"[Rank {ds_engine.global_rank}] Failed to load checkpoint {args.resume_from}, starting from scratch")

# # # for epoch in range(latest_epoch, args.epochs):
# # #     print(f"Entering epoch {epoch+1}", flush=True)

# # #     for i, block in enumerate(train_blocks):
# # #         # Move input block to the correct device and add batch dimension
# # #         inputs = block.unsqueeze(0).to(ds_engine.device)
# # #         labels = inputs.clone()

# # #         outputs = ds_engine(inputs, labels=labels)
# # #         loss = outputs.loss
# # #         ds_engine.backward(loss)
# # #         ds_engine.step()

# # #         if i % 50 == 0 and ds_engine.global_rank == 0:
# # #             print(f"[Rank 0] Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}", flush=True)

# # #         checkpoint_now = (i % 50 == 0) or ((i + 1) % save_interval == 0) or (i + 1 == len(train_blocks))
# # #         if checkpoint_now:
# # #             print(f"[Rank {ds_engine.global_rank}] Saving checkpoint at epoch {epoch+1}, step {i}", flush=True)
# # #             save_start = time.time()
# # #             ds_engine.save_checkpoint(ckpt_dir, tag=f"epoch_{epoch+1}")
# # #             save_time = time.time() - save_start
# # #             if ds_engine.global_rank == 0:
# # #                 print(f"[Rank 0] Checkpoint saved in {save_time:.2f}s", flush=True)

# # import os
# # import time
# # import csv
# # import json
# # from datetime import datetime
# # import argparse

# # import torch
# # import deepspeed
# # from transformers import AutoModelForCausalLM, AutoTokenizer


# # # ----------------------------------------------------
# # # Rank 0 전용 print
# # # ----------------------------------------------------
# # def r0_print(msg):
# #     if os.environ.get("RANK", "0") == "0":
# #         print(msg, flush=True)


# # # ----------------------------------------------------
# # # CSV 로그 초기화 (DataStates 스타일)
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
# #                 "enqueue_seconds",
# #                 "flush_barrier_seconds",

# #                 "model_gpu2cpu_seconds",
# #                 "model_cpu2disk_seconds",
# #                 "optim_gpu2cpu_seconds",
# #                 "optim_cpu2disk_seconds",

# #                 "model_tensor_elements",
# #                 "optim_tensor_elements",

# #                 "model_logical_bytes",
# #                 "optim_logical_bytes",

# #                 "model_throughput_GBps",
# #                 "optim_throughput_GBps",

# #                 "backward_seconds",
# #                 "step_seconds",
# #             ])


# # def append_csv_row(csv_path, row):
# #     with open(csv_path, "a", newline="") as f:
# #         csv.writer(f).writerow(row)


# # # ----------------------------------------------------
# # # 텍스트 파일 → 토큰 블록으로 변환
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

# #             # buffer에 쌓인 토큰이 block_size 이상이면 잘라서 block 생성
# #             while sum(b.numel() for b in buffer) >= block_size:
# #                 concat = torch.cat(buffer)
# #                 blocks.append(concat[:block_size])
# #                 buffer = [concat[block_size:]] if concat.numel() > block_size else []

# #     # 남은 토큰 처리 (padding)
# #     if buffer:
# #         concat = torch.cat(buffer)
# #         pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
# #         if concat.numel() < block_size:
# #             concat = torch.nn.functional.pad(
# #                 concat, (0, block_size - concat.numel()), value=pad_id
# #             )
# #         blocks.append(concat)

# #     return blocks


# # # ----------------------------------------------------
# # # MAIN
# # # ----------------------------------------------------
# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument(
# #         '--deepspeed_config',
# #         type=str,
# #         required=True,
# #         help='Path to DeepSpeed config file',
# #     )
# #     parser.add_argument(
# #         '--model_name_or_path',
# #         type=str,
# #         default='bigscience/bloom-3b',
# #         help='HuggingFace model name or path',
# #     )
# #     parser.add_argument(
# #         '--train_file',
# #         type=str,
# #         required=True,
# #         help='Path to a plain text training file',
# #     )
# #     parser.add_argument(
# #         '--output_dir',
# #         type=str,
# #         default='./bloom3b-finetuned',
# #     )
# #     parser.add_argument(
# #         '--local_rank',
# #         type=int,
# #         default=-1,
# #         help='Local rank passed by deepspeed/torch.distributed',
# #     )
# #     parser.add_argument(
# #         '--per_device_train_batch_size',
# #         type=int,
# #         default=1,
# #         help='Batch size per device/GPU (실제 배치는 DeepSpeed config가 결정)',
# #     )
# #     parser.add_argument(
# #         '--epochs',
# #         type=int,
# #         default=1,
# #         help='Number of training epochs',
# #     )
# #     parser.add_argument(
# #         '--master_port',
# #         type=int,
# #         default=None,
# #         help='Master port for distributed training (optional)',
# #     )
# #     parser.add_argument(
# #         '--resume_from',
# #         type=str,
# #         default=None,
# #         help="Resume training from this checkpoint tag (e.g., 'epoch_3'). If not set, start from scratch.",
# #     )
# #     parser.add_argument(
# #         "--required_steps",
# #         type=int,
# #         default=1070,
# #         help="Total steps per epoch (dataset blocks will be repeated as needed).",
# #     )
# #     parser.add_argument(
# #         "--block_size",
# #         type=int,
# #         default=512,
# #         help="Token block size.",
# #     )

# #     args = parser.parse_args()

# #     print(
# #         f"[boot] env RANK={os.environ.get('RANK')} "
# #         f"LOCAL_RANK(env)={os.environ.get('LOCAL_RANK')} "
# #         f"local_rank(arg)={args.local_rank}",
# #         flush=True,
# #     )

# #     # ---------------------------
# #     # DeepSpeed config 파일 읽기
# #     # ---------------------------
# #     with open(args.deepspeed_config, "r") as f:
# #         ds_config_dict = json.load(f)
# #     r0_print(f"[DEBUG] Loaded DeepSpeed config from {args.deepspeed_config}")

# #     # LOCAL_RANK 우선 사용
# #     args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
# #     if args.local_rank is not None and args.local_rank >= 0:
# #         torch.cuda.set_device(args.local_rank)
# #         print(f"[DEBUG] set_device({args.local_rank})", flush=True)

# #     # ---------------------------
# #     # 모델 / 토크나이저 로드
# #     # ---------------------------
# #     r0_print(f"[DEBUG] Loading model: {args.model_name_or_path}")
# #     model = AutoModelForCausalLM.from_pretrained(
# #         args.model_name_or_path,
# #         device_map=None,
# #         low_cpu_mem_usage=True,
# #         torch_dtype=torch.float16,  # fp16 weight
# #     )
# #     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

# #     # ---------------------------
# #     # 데이터 준비: base_blocks → required_steps까지 반복
# #     # ---------------------------
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

# #     r0_print(f"[Rank 0] base_blocks={len(base_blocks)}, final steps={len(train_blocks)}")

# #     # ------------------------------------------------
# #     # 🔧 Optimizer: torch AdamW 직접 생성해서 DeepSpeed에 넘김
# #     #   → fused_adam CUDA JIT (nvcc 필요) 피하기
# #     # ------------------------------------------------
# #     opt_cfg = ds_config_dict.get("optimizer", {})
# #     opt_params = opt_cfg.get("params", {})

# #     lr = opt_params.get("lr", 2e-5)
# #     betas = tuple(opt_params.get("betas", [0.9, 0.999]))
# #     eps = opt_params.get("eps", 1e-8)
# #     weight_decay = opt_params.get("weight_decay", 0.01)

# #     r0_print(
# #         f"[DEBUG] Building torch.optim.AdamW(lr={lr}, betas={betas}, "
# #         f"eps={eps}, weight_decay={weight_decay})"
# #     )
# #     optimizer = torch.optim.AdamW(
# #         model.parameters(),
# #         lr=lr,
# #         betas=betas,
# #         eps=eps,
# #         weight_decay=weight_decay,
# #     )

# #     # DeepSpeed config에서 optimizer 섹션 제거
# #     # (우리가 직접 optimizer를 넘기므로 DS가 fused_adam 만들려고 안 함)
# #     if "optimizer" in ds_config_dict:
# #         del ds_config_dict["optimizer"]
# #         r0_print("[DEBUG] Removed 'optimizer' from DeepSpeed config (using external AdamW).")

# #     # ------------------------------------------------
# #     # DeepSpeed 초기화
# #     # ------------------------------------------------
# #     ds_engine, _, _, _ = deepspeed.initialize(
# #         model=model,
# #         optimizer=optimizer,
# #         model_parameters=model.parameters(),
# #         config=ds_config_dict,
# #     )
# #     r0_print(f"[DEBUG] DeepSpeed initialized (global_rank={ds_engine.global_rank})")

# #     model.train()

# #     # ------------------------------------------------
# #     # Model logical size 계산 (rank 0 기준)
# #     # ------------------------------------------------
# #     MODEL_TENSOR_ELEMS = 0
# #     MODEL_LOGICAL_BYTES = 0
# #     OPTIM_TENSOR_ELEMS = 0
# #     OPTIM_LOGICAL_BYTES = 0

# #     if ds_engine.global_rank == 0:
# #         params = list(ds_engine.module.parameters())
# #         if len(params) == 0:
# #             raise RuntimeError("No parameters found in model.")

# #         MODEL_TENSOR_ELEMS = sum(p.numel() for p in params)
# #         sample_dtype = params[0].dtype
# #         if sample_dtype in (torch.float16, torch.bfloat16):
# #             bytes_per_elem = 2
# #         else:
# #             bytes_per_elem = 4
# #         MODEL_LOGICAL_BYTES = MODEL_TENSOR_ELEMS * bytes_per_elem

# #         r0_print(
# #             f"[Rank 0] Model logical size ≈ {MODEL_LOGICAL_BYTES / 1e9:.3f} GB "
# #             f"({MODEL_TENSOR_ELEMS} elements, dtype={sample_dtype})"
# #         )

# #     # ------------------------------------------------
# #     # 체크포인트 디렉토리 / CSV
# #     # ------------------------------------------------
# #     ckpt_dir = os.path.join(args.output_dir, "checkpoints")
# #     if ds_engine.global_rank == 0:
# #         os.makedirs(ckpt_dir, exist_ok=True)
# #         metrics_csv = os.path.join(ckpt_dir, "metrics_log.csv")
# #         init_csv_log(metrics_csv)
# #     else:
# #         metrics_csv = None

# #     # ------------------------------------------------
# #     # Resume logic
# #     # ------------------------------------------------
# #     latest_epoch = 0
# #     if args.resume_from is not None:
# #         print(
# #             f"[Rank {ds_engine.global_rank}] Attempting to resume from checkpoint: {args.resume_from}",
# #             flush=True,
# #         )
# #         load_start = time.time()
# #         load_ret = ds_engine.load_checkpoint(ckpt_dir, tag=args.resume_from)
# #         load_time = time.time() - load_start

# #         if isinstance(load_ret, tuple):
# #             load_success = load_ret[0]
# #             client_state = load_ret[1] if len(load_ret) > 1 else None
# #         else:
# #             load_success = bool(load_ret)
# #             client_state = None

# #         if ds_engine.global_rank == 0:
# #             print(f"[Rank 0] Checkpoint loaded in {load_time:.2f}s", flush=True)

# #         if load_success:
# #             try:
# #                 latest_epoch = int(args.resume_from.split("_")[1])
# #             except Exception:
# #                 latest_epoch = 0
# #             print(
# #                 f"[Rank {ds_engine.global_rank}] Successfully resumed from {args.resume_from}, "
# #                 f"continuing at epoch {latest_epoch+1}",
# #                 flush=True,
# #             )
# #         else:
# #             print(
# #                 f"[Rank {ds_engine.global_rank}] Failed to load checkpoint {args.resume_from}, starting from scratch",
# #                 flush=True,
# #             )

# #     # ------------------------------------------------
# #     # TRAINING LOOP
# #     # ------------------------------------------------
# #     save_interval = 200
# #     r0_print(
# #         f"[Rank 0] Epochs={args.epochs}, save_interval={save_interval}, "
# #         f"steps/epoch={REQUIRED_STEPS}"
# #     )

# #     for epoch in range(latest_epoch, args.epochs):
# #         r0_print(f"[Rank 0] Entering epoch {epoch+1}")

# #         for step, block in enumerate(train_blocks):

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
# #             #   (gradient_accumulation_steps는 내부에서 처리)
# #             # -------------------------
# #             t_step_start = time.time()
# #             ds_engine.step()
# #             t_step_end = time.time()

# #             if step % 50 == 0 and ds_engine.global_rank == 0:
# #                 print(
# #                     f"[Rank 0] Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}",
# #                     flush=True,
# #                 )

# #             # -------------------------
# #             # Checkpoint 조건
# #             # -------------------------
# #             is_last = (step + 1 == REQUIRED_STEPS)
# #             checkpoint_now = (step % save_interval == 0) or is_last

# #             if checkpoint_now:
# #                 tag = f"epoch_{epoch+1}" if is_last else f"step_{step+1}"
# #                 print(
# #                     f"[Rank {ds_engine.global_rank}] Saving checkpoint at {tag}",
# #                     flush=True,
# #                 )

# #                 t_ckpt_begin = time.time()
# #                 ds_engine.save_checkpoint(ckpt_dir, tag=tag)
# #                 t_ckpt_end = time.time()

# #                 total_ckpt_seconds = t_ckpt_end - t_ckpt_begin

# #                 # DeepSpeed 기본 save는 동기형으로 가정
# #                 enqueue_seconds = total_ckpt_seconds
# #                 flush_barrier_seconds = 0.0

# #                 # 내부 세부 시간은 알 수 없으니 0으로 기록
# #                 model_gpu2cpu_seconds = 0.0
# #                 model_cpu2disk_seconds = 0.0
# #                 optim_gpu2cpu_seconds = 0.0
# #                 optim_cpu2disk_seconds = 0.0

# #                 if ds_engine.global_rank == 0:
# #                     model_tensor_elements = MODEL_TENSOR_ELEMS
# #                     model_logical_bytes = MODEL_LOGICAL_BYTES
# #                     optim_tensor_elements = OPTIM_TENSOR_ELEMS
# #                     optim_logical_bytes = OPTIM_LOGICAL_BYTES

# #                     model_throughput = 0.0
# #                     if total_ckpt_seconds > 0 and model_logical_bytes > 0:
# #                         model_throughput = (
# #                             model_logical_bytes / total_ckpt_seconds
# #                         ) / 1e9

# #                     optim_throughput = 0.0

# #                     backward_seconds = (t_bw_end - t_bw_start)
# #                     step_seconds = (t_step_end - t_step_start)

# #                     append_csv_row(metrics_csv, [
# #                         datetime.now().isoformat(),
# #                         epoch + 1,
# #                         step,
# #                         loss.item(),
# #                         ds_engine.global_rank,

# #                         total_ckpt_seconds,
# #                         enqueue_seconds,
# #                         flush_barrier_seconds,

# #                         model_gpu2cpu_seconds,
# #                         model_cpu2disk_seconds,
# #                         optim_gpu2cpu_seconds,
# #                         optim_cpu2disk_seconds,

# #                         model_tensor_elements,
# #                         optim_tensor_elements,

# #                         model_logical_bytes,
# #                         optim_logical_bytes,

# #                         model_throughput,
# #                         optim_throughput,

# #                         backward_seconds,
# #                         step_seconds,
# #                     ])

# #                     print(
# #                         f"[Rank 0][CKPT] {tag}: total={total_ckpt_seconds:.3f}s, "
# #                         f"model_size≈{model_logical_bytes/1e9:.3f} GB, "
# #                         f"model_throughput≈{model_throughput:.3f} GB/s",
# #                         flush=True,
# #                     )

# #     r0_print("[Rank 0] Training finished.")


# # if __name__ == "__main__":
# #     main()

# import os
# import time
# import csv
# import json
# from datetime import datetime
# import argparse

# import torch
# import deepspeed
# from transformers import AutoModelForCausalLM, AutoTokenizer


# # ----------------------------------------------------
# # Rank 0 only print (for debug)
# # ----------------------------------------------------
# def r0_print(msg: str):
#     if os.environ.get("RANK", "0") == "0":
#         print(msg, flush=True)


# # ----------------------------------------------------
# # CSV logger: DataStates 스타일 헤더
# # ----------------------------------------------------
# def init_csv_log(csv_path: str):
#     if not os.path.exists(csv_path):
#         with open(csv_path, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 "timestamp",
#                 "epoch",
#                 "step",
#                 "loss",
#                 "rank",

#                 "total_ckpt_seconds",
#                 "enqueue_seconds",
#                 "flush_barrier_seconds",

#                 "model_gpu2cpu_seconds",
#                 "model_cpu2disk_seconds",
#                 "optim_gpu2cpu_seconds",
#                 "optim_cpu2disk_seconds",

#                 "model_tensor_elements",
#                 "optim_tensor_elements",

#                 "model_logical_bytes",
#                 "optim_logical_bytes",

#                 "model_throughput_GBps",
#                 "optim_throughput_GBps",

#                 "backward_seconds",
#                 "step_seconds",
#             ])


# def append_csv_row(csv_path: str, row):
#     with open(csv_path, "a", newline="") as f:
#         csv.writer(f).writerow(row)


# # ----------------------------------------------------
# # Text file → token blocks
# # ----------------------------------------------------
# def load_dataset_stream(file_path, tokenizer, block_size=512):
#     """
#     train_file을 읽어서, 길이 block_size 인 1D tensor block 리스트로 변환.
#     """
#     blocks = []
#     buffer = []

#     r0_print(f"[DEBUG] Loading train file: {file_path}")

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

#             # buffer에 쌓인 토큰 합이 block_size 이상이면 잘라서 block 하나 생성
#             while sum(b.numel() for b in buffer) >= block_size:
#                 concat = torch.cat(buffer)
#                 blocks.append(concat[:block_size])
#                 buffer = [concat[block_size:]] if concat.numel() > block_size else []

#     # 남은 토큰 처리 (마지막 block, padding)
#     if buffer:
#         concat = torch.cat(buffer)
#         pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
#         if concat.numel() < block_size:
#             concat = torch.nn.functional.pad(
#                 concat, (0, block_size - concat.numel()), value=pad_id
#             )
#         blocks.append(concat)

#     r0_print(f"[DEBUG] Loaded {len(blocks)} blocks from dataset.")
#     return blocks


# # ----------------------------------------------------
# # MAIN
# # ----------------------------------------------------
# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--deepspeed_config",
#         type=str,
#         required=True,
#         help="Path to DeepSpeed config JSON (e.g., ds_config_zero2.json)",
#     )
#     parser.add_argument(
#         "--model_name_or_path",
#         type=str,
#         required=True,
#         help="HuggingFace model name or path (e.g., bigscience/bloom-3b)",
#     )
#     parser.add_argument(
#         "--train_file",
#         type=str,
#         required=True,
#         help="Plain text training file (one sample per line, etc.)",
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="./bloom_finetuned",
#     )
#     parser.add_argument(
#         "--local_rank",
#         type=int,
#         default=-1,
#         help="Local rank passed by DeepSpeed/torch.distributed",
#     )
#     parser.add_argument(
#         "--epochs",
#         type=int,
#         default=1,
#     )
#     parser.add_argument(
#         "--required_steps",
#         type=int,
#         default=1070,
#         help="Total global steps per epoch (dataset will be repeated).",
#     )
#     parser.add_argument(
#         "--block_size",
#         type=int,
#         default=512,
#     )
#     parser.add_argument(
#         "--resume_from",
#         type=str,
#         default=None,
#         help="Checkpoint tag to resume from (e.g., epoch_1_step_200).",
#     )

#     args = parser.parse_args()

#     # --------------------------
#     # Debug: rank info
#     # --------------------------
#     print(
#         f"[boot] RANK={os.environ.get('RANK')} "
#         f"LOCAL_RANK(env)={os.environ.get('LOCAL_RANK')} "
#         f"local_rank(arg)={args.local_rank}",
#         flush=True,
#     )

#     # --------------------------
#     # Load DeepSpeed config JSON
#     # --------------------------
#     with open(args.deepspeed_config, "r") as f:
#         ds_config = json.load(f)
#     r0_print(f"[DEBUG] Loaded DeepSpeed config from: {args.deepspeed_config}")

#     # --------------------------
#     # local_rank / device 설정
#     # --------------------------
#     args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
#     if args.local_rank is not None and args.local_rank >= 0:
#         torch.cuda.set_device(args.local_rank)
#         print(f"[DEBUG] torch.cuda.set_device({args.local_rank})", flush=True)

#     # --------------------------
#     # 모델 / 토크나이저 로드
#     # --------------------------
#     r0_print(f"[DEBUG] Loading model: {args.model_name_or_path}")
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_name_or_path,
#         device_map=None,           # DeepSpeed가 관리
#         low_cpu_mem_usage=True,
#         torch_dtype=torch.float16, # fp16 weights
#     )
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

#     # --------------------------
#     # 데이터 준비: base_blocks → required_steps까지 반복
#     # --------------------------
#     base_blocks = load_dataset_stream(
#         args.train_file, tokenizer, block_size=args.block_size
#     )
#     if len(base_blocks) == 0:
#         raise RuntimeError("Dataset produced 0 blocks — training impossible.")

#     REQUIRED_STEPS = args.required_steps
#     repeated = []
#     while len(repeated) < REQUIRED_STEPS:
#         repeated.extend(base_blocks)
#     train_blocks = repeated[:REQUIRED_STEPS]

#     r0_print(
#         f"[Rank 0] base_blocks={len(base_blocks)}, "
#         f"final steps (train_blocks)={len(train_blocks)}"
#     )

#     # --------------------------
#     # Optimizer: torch.optim.AdamW 직접 생성
#     #   → DeepSpeed fused_adam / nvcc 컴파일 피하기
#     # --------------------------
#     opt_cfg = ds_config.get("optimizer", {})
#     opt_params = opt_cfg.get("params", {})

#     lr = opt_params.get("lr", 2e-5)
#     betas = tuple(opt_params.get("betas", [0.9, 0.999]))
#     eps = opt_params.get("eps", 1e-8)
#     weight_decay = opt_params.get("weight_decay", 0.01)

#     r0_print(
#         f"[DEBUG] Building torch.optim.AdamW("
#         f"lr={lr}, betas={betas}, eps={eps}, weight_decay={weight_decay})"
#     )
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=lr,
#         betas=betas,
#         eps=eps,
#         weight_decay=weight_decay,
#     )

#     # DeepSpeed가 config 기반 optimizer(FusedAdam 등) 만들지 않도록 제거
#     if "optimizer" in ds_config:
#         del ds_config["optimizer"]
#         r0_print("[DEBUG] Removed 'optimizer' from DeepSpeed config (using external AdamW).")

#     # --------------------------
#     # DeepSpeed initialize
#     # --------------------------
#     engine, _, _, _ = deepspeed.initialize(
#         model=model,
#         optimizer=optimizer,
#         model_parameters=model.parameters(),
#         config=ds_config,   # dict로 전달
#     )

#     r0_print(f"[DEBUG] DeepSpeed initialized: global_rank={engine.global_rank}")

#     # --------------------------
#     # Model logical size (rank 0)
#     # --------------------------
#     MODEL_TENSOR_ELEMS = 0
#     MODEL_LOGICAL_BYTES = 0
#     OPTIM_TENSOR_ELEMS = 0
#     OPTIM_LOGICAL_BYTES = 0

#     if engine.global_rank == 0:
#         params = list(engine.module.parameters())
#         if len(params) == 0:
#             raise RuntimeError("No parameters found in model.")

#         MODEL_TENSOR_ELEMS = sum(p.numel() for p in params)
#         sample_dtype = params[0].dtype

#         if sample_dtype in (torch.float16, torch.bfloat16):
#             bytes_per_elem = 2
#         else:
#             bytes_per_elem = 4

#         MODEL_LOGICAL_BYTES = MODEL_TENSOR_ELEMS * bytes_per_elem

#         r0_print(
#             f"[Rank 0] Model logical size ≈ {MODEL_LOGICAL_BYTES / 1e9:.3f} GB "
#             f"({MODEL_TENSOR_ELEMS} elements, dtype={sample_dtype})"
#         )

#     # --------------------------
#     # Checkpoint dir & CSV
#     # --------------------------
#     ckpt_dir = os.path.join(args.output_dir, "checkpoints")
#     if engine.global_rank == 0:
#         os.makedirs(ckpt_dir, exist_ok=True)
#         metrics_csv = os.path.join(ckpt_dir, "metrics_log.csv")
#         init_csv_log(metrics_csv)
#         print(f"[Rank 0] Checkpoints & CSV under: {ckpt_dir}", flush=True)
#     else:
#         metrics_csv = None

#     # --------------------------
#     # (선택) resume
#     # --------------------------
#     latest_epoch = 0
#     if args.resume_from is not None:
#         print(
#             f"[Rank {engine.global_rank}] Trying to resume from tag={args.resume_from}",
#             flush=True,
#         )
#         t0 = time.time()
#         load_ret = engine.load_checkpoint(ckpt_dir, tag=args.resume_from)
#         load_time = time.time() - t0

#         if isinstance(load_ret, tuple):
#             load_success = load_ret[0]
#         else:
#             load_success = bool(load_ret)

#         if engine.global_rank == 0:
#             print(f"[Rank 0] load_checkpoint time={load_time:.2f}s", flush=True)

#         if load_success:
#             print(
#                 f"[Rank {engine.global_rank}] Successfully resumed from {args.resume_from}",
#                 flush=True,
#             )
#             # tag에서 epoch 숫자 파싱할 수도 있지만, 여기서는 0으로 두고 계속
#         else:
#             print(
#                 f"[Rank {engine.global_rank}] Failed to resume from {args.resume_from}, starting fresh",
#                 flush=True,
#             )

#     # --------------------------
#     # Training loop
#     # --------------------------
#     save_interval = 200   # DataStates 실험이랑 비슷하게 200 step마다 ckpt
#     engine.train()

#     r0_print(
#         f"[Rank 0] Start training: epochs={args.epochs}, "
#         f"steps_per_epoch={REQUIRED_STEPS}, save_interval={save_interval}"
#     )

#     for epoch in range(latest_epoch, args.epochs):
#         if engine.global_rank == 0:
#             print(f"[Rank 0] ===== Entering epoch {epoch + 1} =====", flush=True)

#         for step, block in enumerate(train_blocks):
#             # ---- Forward + Backward ----
#             t_bw_start = time.time()

#             inputs = block.unsqueeze(0).to(engine.device)
#             labels = inputs.clone()

#             outputs = engine(inputs, labels=labels)
#             loss = outputs.loss

#             engine.backward(loss)
#             t_bw_end = time.time()

#             # ---- Optimizer step ----
#             t_step_start = time.time()
#             engine.step()
#             t_step_end = time.time()

#             backward_seconds = t_bw_end - t_bw_start
#             step_seconds = t_step_end - t_step_start

#             # 디버그 출력
#             if step % 50 == 0 and engine.global_rank == 0:
#                 print(
#                     f"[Rank 0] epoch={epoch+1}/{args.epochs} "
#                     f"step={step}/{len(train_blocks)} "
#                     f"loss={loss.item():.4f} "
#                     f"bw={backward_seconds:.3f}s "
#                     f"step={step_seconds:.3f}s",
#                     flush=True,
#                 )

#             # ---- Checkpoint timing (DeepSpeed 기본 save_checkpoint) ----
#             is_last_step = (step + 1 == len(train_blocks))
#             do_ckpt = (step % save_interval == 0) or is_last_step

#             if do_ckpt:
#                 tag = f"epoch_{epoch+1}_step_{step+1}"

#                 if engine.global_rank == 0:
#                     print(f"[Rank 0] Saving checkpoint: tag={tag}", flush=True)

#                 t_ckpt_begin = time.time()
#                 engine.save_checkpoint(ckpt_dir, tag=tag)
#                 t_ckpt_end = time.time()

#                 total_ckpt_seconds = t_ckpt_end - t_ckpt_begin

#                 # DeepSpeed baseline은 sync 저장이므로 이렇게 기록:
#                 enqueue_seconds = total_ckpt_seconds
#                 flush_barrier_seconds = 0.0

#                 # GPU→CPU, CPU→DISK 세부 분해는 불가 → 0으로 기록
#                 model_gpu2cpu_seconds = 0.0
#                 model_cpu2disk_seconds = 0.0
#                 optim_gpu2cpu_seconds = 0.0
#                 optim_cpu2disk_seconds = 0.0

#                 if engine.global_rank == 0:
#                     model_tensor_elements = MODEL_TENSOR_ELEMS
#                     model_logical_bytes = MODEL_LOGICAL_BYTES
#                     optim_tensor_elements = OPTIM_TENSOR_ELEMS
#                     optim_logical_bytes = OPTIM_LOGICAL_BYTES

#                     model_throughput = 0.0
#                     if total_ckpt_seconds > 0 and model_logical_bytes > 0:
#                         model_throughput = (
#                             model_logical_bytes / total_ckpt_seconds
#                         ) / 1e9

#                     optim_throughput = 0.0  # 여기서는 0

#                     append_csv_row(metrics_csv, [
#                         datetime.now().isoformat(),
#                         epoch + 1,
#                         step,
#                         loss.item(),
#                         engine.global_rank,

#                         total_ckpt_seconds,
#                         enqueue_seconds,
#                         flush_barrier_seconds,

#                         model_gpu2cpu_seconds,
#                         model_cpu2disk_seconds,
#                         optim_gpu2cpu_seconds,
#                         optim_cpu2disk_seconds,

#                         model_tensor_elements,
#                         optim_tensor_elements,

#                         model_logical_bytes,
#                         optim_logical_bytes,

#                         model_throughput,
#                         optim_throughput,

#                         backward_seconds,
#                         step_seconds,
#                     ])

#                     print(
#                         f"[Rank 0][CKPT] tag={tag} "
#                         f"total={total_ckpt_seconds:.3f}s, "
#                         f"model_size≈{model_logical_bytes/1e9:.3f} GB, "
#                         f"model_throughput≈{model_throughput:.3f} GB/s",
#                         flush=True,
#                     )

#     r0_print("[Rank 0] Training finished.")


# if __name__ == "__main__":
#     main()

import os
import time
import csv
import json
from datetime import datetime
import argparse

import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------------------------------
# Rank 0 only print (for debug)
# ----------------------------------------------------
def r0_print(msg: str):
    if os.environ.get("RANK", "0") == "0":
        print(msg, flush=True)


# ----------------------------------------------------
# CSV logger: DataStates 스타일 헤더
# ----------------------------------------------------
def init_csv_log(csv_path: str):
    """
    DeepSpeed 실험용 메트릭 CSV 초기화.
    DataStates와 비슷한 스키마 + has_checkpoint 플래그 추가.
    """
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "epoch",
                "step",
                "loss",
                "rank",

                # checkpoint timing
                "has_checkpoint",          # 1: this step saved ckpt, 0: no ckpt
                "total_ckpt_seconds",
                "enqueue_seconds",
                "flush_barrier_seconds",

                # DeepSpeed는 breakdown 없음 → 0으로 유지
                "model_gpu2cpu_seconds",
                "model_cpu2disk_seconds",
                "optim_gpu2cpu_seconds",
                "optim_cpu2disk_seconds",

                # logical size & throughput
                "model_tensor_elements",
                "optim_tensor_elements",
                "model_logical_bytes",
                "optim_logical_bytes",
                "model_throughput_GBps",
                "optim_throughput_GBps",

                # training time
                "backward_seconds",
                "step_seconds",
            ])


def append_csv_row(csv_path: str, row):
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ----------------------------------------------------
# Text file → token blocks
# ----------------------------------------------------
def load_dataset_stream(file_path, tokenizer, block_size=512):
    """
    train_file을 읽어서, 길이 block_size 인 1D tensor block 리스트로 변환.
    """
    blocks = []
    buffer = []

    r0_print(f"[DEBUG] Loading train file: {file_path}")

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

            # buffer에 쌓인 토큰 합이 block_size 이상이면 잘라서 block 하나 생성
            while sum(b.numel() for b in buffer) >= block_size:
                concat = torch.cat(buffer)
                blocks.append(concat[:block_size])
                buffer = [concat[block_size:]] if concat.numel() > block_size else []

    # 남은 토큰 처리 (마지막 block, padding)
    if buffer:
        concat = torch.cat(buffer)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        if concat.numel() < block_size:
            concat = torch.nn.functional.pad(
                concat, (0, block_size - concat.numel()), value=pad_id
            )
        blocks.append(concat)

    r0_print(f"[DEBUG] Loaded {len(blocks)} blocks from dataset.")
    return blocks


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--deepspeed_config",
        type=str,
        required=True,
        help="Path to DeepSpeed config JSON (e.g., ds_config_zero2.json)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="HuggingFace model name or path (e.g., bigscience/bloom-3b)",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Plain text training file (one sample per line, etc.)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./bloom_finetuned",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank passed by DeepSpeed/torch.distributed",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--required_steps",
        type=int,
        default=1070,
        help="Total global steps per epoch (dataset will be repeated).",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Checkpoint tag to resume from (e.g., epoch_1_step_200).",
    )

    args = parser.parse_args()

    # --------------------------
    # Debug: rank info
    # --------------------------
    print(
        f"[boot] RANK={os.environ.get('RANK')} "
        f"LOCAL_RANK(env)={os.environ.get('LOCAL_RANK')} "
        f"local_rank(arg)={args.local_rank}",
        flush=True,
    )

    # --------------------------
    # Load DeepSpeed config JSON
    # --------------------------
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    r0_print(f"[DEBUG] Loaded DeepSpeed config from: {args.deepspeed_config}")

    # --------------------------
    # local_rank / device 설정
    # --------------------------
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if args.local_rank is not None and args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        print(f"[DEBUG] torch.cuda.set_device({args.local_rank})", flush=True)

    # --------------------------
    # 모델 / 토크나이저 로드
    # --------------------------
    r0_print(f"[DEBUG] Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=None,           # DeepSpeed가 관리
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16, # fp16 weights
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # --------------------------
    # 데이터 준비: base_blocks → required_steps까지 반복
    # --------------------------
    base_blocks = load_dataset_stream(
        args.train_file, tokenizer, block_size=args.block_size
    )
    if len(base_blocks) == 0:
        raise RuntimeError("Dataset produced 0 blocks — training impossible.")

    REQUIRED_STEPS = args.required_steps
    repeated = []
    while len(repeated) < REQUIRED_STEPS:
        repeated.extend(base_blocks)
    train_blocks = repeated[:REQUIRED_STEPS]

    r0_print(
        f"[Rank 0] base_blocks={len(base_blocks)}, "
        f"final steps (train_blocks)={len(train_blocks)}"
    )

    # --------------------------
    # Optimizer: torch.optim.AdamW 직접 생성
    #   → DeepSpeed fused_adam / nvcc 컴파일 피하기
    # --------------------------
    opt_cfg = ds_config.get("optimizer", {})
    opt_params = opt_cfg.get("params", {})

    lr = opt_params.get("lr", 2e-5)
    betas = tuple(opt_params.get("betas", [0.9, 0.999]))
    eps = opt_params.get("eps", 1e-8)
    weight_decay = opt_params.get("weight_decay", 0.01)

    r0_print(
        f"[DEBUG] Building torch.optim.AdamW("
        f"lr={lr}, betas={betas}, eps={eps}, weight_decay={weight_decay})"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    # DeepSpeed가 config 기반 optimizer(FusedAdam 등) 만들지 않도록 제거
    if "optimizer" in ds_config:
        del ds_config["optimizer"]
        r0_print("[DEBUG] Removed 'optimizer' from DeepSpeed config (using external AdamW).")

    # --------------------------
    # DeepSpeed initialize
    # --------------------------
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        config=ds_config,   # dict로 전달
    )

    r0_print(f"[DEBUG] DeepSpeed initialized: global_rank={engine.global_rank}")

    # --------------------------
    # Model logical size (rank 0)
    # --------------------------
    MODEL_TENSOR_ELEMS = 0
    MODEL_LOGICAL_BYTES = 0
    OPTIM_TENSOR_ELEMS = 0
    OPTIM_LOGICAL_BYTES = 0

    if engine.global_rank == 0:
        params = list(engine.module.parameters())
        if len(params) == 0:
            raise RuntimeError("No parameters found in model.")

        MODEL_TENSOR_ELEMS = sum(p.numel() for p in params)
        sample_dtype = params[0].dtype

        if sample_dtype in (torch.float16, torch.bfloat16):
            bytes_per_elem = 2
        else:
            bytes_per_elem = 4

        MODEL_LOGICAL_BYTES = MODEL_TENSOR_ELEMS * bytes_per_elem

        r0_print(
            f"[Rank 0] Model logical size ≈ {MODEL_LOGICAL_BYTES / 1e9:.3f} GB "
            f"({MODEL_TENSOR_ELEMS} elements, dtype={sample_dtype})"
        )

    # --------------------------
    # Checkpoint dir & CSV
    # --------------------------
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    if engine.global_rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        metrics_csv = os.path.join(ckpt_dir, "metrics_log.csv")
        init_csv_log(metrics_csv)
        print(f"[Rank 0] Checkpoints & CSV under: {ckpt_dir}", flush=True)
    else:
        metrics_csv = None

    # --------------------------
    # (선택) resume
    # --------------------------
    latest_epoch = 0
    if args.resume_from is not None:
        print(
            f"[Rank {engine.global_rank}] Trying to resume from tag={args.resume_from}",
            flush=True,
        )
        t0 = time.time()
        load_ret = engine.load_checkpoint(ckpt_dir, tag=args.resume_from)
        load_time = time.time() - t0

        if isinstance(load_ret, tuple):
            load_success = load_ret[0]
        else:
            load_success = bool(load_ret)

        if engine.global_rank == 0:
            print(f"[Rank 0] load_checkpoint time={load_time:.2f}s", flush=True)

        if load_success:
            print(
                f"[Rank {engine.global_rank}] Successfully resumed from {args.resume_from}",
                flush=True,
            )
        else:
            print(
                f"[Rank {engine.global_rank}] Failed to resume from {args.resume_from}, starting fresh",
                flush=True,
            )

    # --------------------------
    # Training loop
    # --------------------------
    save_interval = 200   # DataStates 실험이랑 비슷하게 200 step마다 ckpt
    engine.train()

    r0_print(
        f"[Rank 0] Start training: epochs={args.epochs}, "
        f"steps_per_epoch={REQUIRED_STEPS}, save_interval={save_interval}"
    )

    for epoch in range(latest_epoch, args.epochs):
        if engine.global_rank == 0:
            print(f"[Rank 0] ===== Entering epoch {epoch + 1} =====", flush=True)

        for step, block in enumerate(train_blocks):
            # ---- Forward + Backward ----
            t_bw_start = time.time()

            inputs = block.unsqueeze(0).to(engine.device)
            labels = inputs.clone()

            outputs = engine(inputs, labels=labels)
            loss = outputs.loss

            engine.backward(loss)
            t_bw_end = time.time()

            # ---- Optimizer step ----
            t_step_start = time.time()
            engine.step()
            t_step_end = time.time()

            backward_seconds = t_bw_end - t_bw_start
            step_seconds = t_step_end - t_step_start

            # 디버그 출력
            if step % 50 == 0 and engine.global_rank == 0:
                print(
                    f"[Rank 0] epoch={epoch+1}/{args.epochs} "
                    f"step={step}/{len(train_blocks)} "
                    f"loss={loss.item():.4f} "
                    f"bw={backward_seconds:.3f}s "
                    f"step={step_seconds:.3f}s",
                    flush=True,
                )

            # ---- Checkpoint 여부 결정 ----
            is_last_step = (step + 1 == len(train_blocks))
            do_ckpt = (step % save_interval == 0) or is_last_step

            # 기본값: checkpoint 없음
            has_checkpoint = 0
            total_ckpt_seconds = 0.0
            enqueue_seconds = 0.0
            flush_barrier_seconds = 0.0
            model_gpu2cpu_seconds = 0.0
            model_cpu2disk_seconds = 0.0
            optim_gpu2cpu_seconds = 0.0
            optim_cpu2disk_seconds = 0.0
            model_throughput = 0.0
            optim_throughput = 0.0

            if do_ckpt:
                has_checkpoint = 1
                tag = f"epoch_{epoch+1}_step_{step+1}"

                if engine.global_rank == 0:
                    print(f"[Rank 0] Saving checkpoint: tag={tag}", flush=True)

                t_ckpt_begin = time.time()
                engine.save_checkpoint(ckpt_dir, tag=tag)
                t_ckpt_end = time.time()

                total_ckpt_seconds = t_ckpt_end - t_ckpt_begin

                # DeepSpeed baseline은 sync 저장이므로 이렇게 기록:
                enqueue_seconds = total_ckpt_seconds
                flush_barrier_seconds = 0.0

                if engine.global_rank == 0:
                    model_tensor_elements = MODEL_TENSOR_ELEMS
                    model_logical_bytes = MODEL_LOGICAL_BYTES
                    optim_tensor_elements = OPTIM_TENSOR_ELEMS
                    optim_logical_bytes = OPTIM_LOGICAL_BYTES

                    if total_ckpt_seconds > 0 and model_logical_bytes > 0:
                        model_throughput = (
                            model_logical_bytes / total_ckpt_seconds
                        ) / 1e9

                    optim_throughput = 0.0

                    print(
                        f"[Rank 0][CKPT] tag={tag} "
                        f"total={total_ckpt_seconds:.3f}s, "
                        f"model_size≈{model_logical_bytes/1e9:.3f} GB, "
                        f"model_throughput≈{model_throughput:.3f} GB/s",
                        flush=True,
                    )

            # ---- CSV 기록: 모든 step에 대해 1줄 ----
            if engine.global_rank == 0:
                model_tensor_elements = MODEL_TENSOR_ELEMS
                model_logical_bytes = MODEL_LOGICAL_BYTES
                optim_tensor_elements = OPTIM_TENSOR_ELEMS
                optim_logical_bytes = OPTIM_LOGICAL_BYTES

                append_csv_row(metrics_csv, [
                    datetime.now().isoformat(),
                    epoch + 1,
                    step,
                    loss.item(),
                    engine.global_rank,

                    has_checkpoint,
                    total_ckpt_seconds,
                    enqueue_seconds,
                    flush_barrier_seconds,

                    model_gpu2cpu_seconds,
                    model_cpu2disk_seconds,
                    optim_gpu2cpu_seconds,
                    optim_cpu2disk_seconds,

                    model_tensor_elements,
                    optim_tensor_elements,
                    model_logical_bytes,
                    optim_logical_bytes,
                    model_throughput,
                    optim_throughput,

                    backward_seconds,
                    step_seconds,
                ])

    r0_print("[Rank 0] Training finished.")


if __name__ == "__main__":
    main()