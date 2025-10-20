# CheckFreq + BLOOM-560m (LLM-Checkpoints) — Docker Quickstart

This guide shows how to train **BLOOM-560m** with **CheckFreq** using Docker + PyTorch (CUDA 11.8).  
It includes **single-GPU AUTO**, **multi-GPU MANUAL**, and **multi-GPU AUTO** runs, with clear **Host** vs **Inside container** steps.

> **Important:** Don’t put comments at the end of lines that use `\` (line-continuations). Bash will treat the comment as part of the command and fail.

---

## Prerequisites

- NVIDIA driver installed on the host
- Docker + NVIDIA Container Toolkit
- A workspace directory on the host, e.g. `/home/yi/checkfreq_env` (adjust paths if different)
- Training text at `/home/yi/checkfreq_env/wt2_small.txt` (mounted into the container at `/work/wt2_small.txt`)

---

## A) One-time setup

### (Host)

```bash
# Pull CUDA 11.8 PyTorch runtime
sudo docker pull pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# Start a container (all GPUs). Adjust the left side of -v if your host path differs.
sudo docker run -it --gpus all --ipc=host --shm-size 32g \
  -v /home/yi/checkfreq_env:/work \
  --workdir /work \
  --name cf-bloom \
  pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime bash
```


if you have it,

```bash
# See if it's running or stopped
sudo docker ps -a --filter "name=^/cf-bloom$" --format "table {{.Names}}\t{{.Status}}"

# If it's not running, start it
sudo docker start cf-bloom

# Open a shell inside it
sudo docker exec -it cf-bloom bash
```
### (Inside the container)

```bash
# Tools + quick GPU sanity
apt-get update && apt-get install -y git
nvidia-smi
python - <<'PY'
import torch
print("cuda.is_available:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count())
PY

# Clone repos
cd /work
git clone https://github.com/msr-fiddle/CheckFreq.git
git clone --branch checkfreq_bloom --single-branch https://github.com/OlgaKogiou/LLM-Checkpoints.git
mkdir -p /work/hf_cache

# Python deps (once per container)
python -m pip install -U pip wheel setuptools
pip install --no-cache-dir "transformers<4.46" "datasets<3" "accelerate>=0.30,<0.33" sentencepiece

# Common env (re-export before any run)
export HF_HOME=/work/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/work/LLM-Checkpoints:/work/CheckFreq/src:$PYTHONPATH
export CF_USE_THREAD=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1

# (Optional) verify the DALI-free iterator is used
python - <<'PY'
import cf_iterator, inspect
print("cf_iterator path:", inspect.getsourcefile(cf_iterator))
PY
```

---

## B) Single-GPU (AUTO mode)

### (Inside the container)

```bash
export HF_HOME=/work/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/work/LLM-Checkpoints:/work/CheckFreq/src:$PYTHONPATH
export CF_USE_THREAD=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p /work/chk_bloom_auto_single

python -u /work/LLM-Checkpoints/models/nlp/bloom_cf.py \
  --model bigscience/bloom-560m \
  --train-file /work/wt2_small.txt \
  --seq-len 128 \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --epochs 1 \
  --workers 2 \
  --lr 2e-5 \
  --chk-prefix /work/chk_bloom_auto_single \
  --manual-freq 0 \
  --arch-name bloom560m | tee -a /work/chk_bloom_auto_single/run.log
```

---

## C) Multi-GPU (MANUAL mode, 4 GPUs)

### (Host)

```bash
# Fresh 4-GPU container
sudo docker rm -f checkfreq-dev 2>/dev/null || true
sudo docker run -it --gpus '"device=0,1,2,3"' --ipc=host --shm-size 32g \
  -v /home/yi/checkfreq_env:/work \
  --workdir /work \
  --name checkfreq-dev \
  pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime bash
```

### (Inside the container)

```bash
# (If fresh container) deps + env
apt-get update && apt-get install -y git
cd /work
[ -d CheckFreq ] || git clone https://github.com/msr-fiddle/CheckFreq.git
[ -d LLM-Checkpoints ] || git clone --branch checkfreq_bloom --single-branch https://github.com/OlgaKogiou/LLM-Checkpoints.git
python -m pip install -U pip wheel setuptools
pip install --no-cache-dir "transformers<4.46" "datasets<3" "accelerate>=0.30,<0.33" sentencepiece

export HF_HOME=/work/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/work/LLM-Checkpoints:/work/CheckFreq/src:$PYTHONPATH
export CF_USE_THREAD=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

mkdir -p /work/chk_bloom_multi

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  /work/LLM-Checkpoints/models/nlp/bloom_cf.py \
  --model bigscience/bloom-560m \
  --train-file /work/wt2_small.txt \
  --seq-len 128 \
  --batch-size 2 \
  --grad-accum-steps 4 \
  --epochs 1 \
  --workers 2 \
  --lr 2e-5 \
  --chk-prefix /work/chk_bloom_multi \
  --manual-freq 200 \
  --arch-name bloom560m | tee -a /work/chk_bloom_multi/run.log
```

---

## D) Multi-GPU (AUTO mode, 4 GPUs)

### (Host)

```bash
sudo docker rm -f checkfreq-dev 2>/dev/null || true
sudo docker run -it --gpus '"device=0,1,2,3"' --ipc=host --shm-size 32g \
  -v /home/yi/checkfreq_env:/work \
  --workdir /work \
  --name checkfreq-dev \
  pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime bash
```

### (Inside the container)

```bash
export HF_HOME=/work/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/work/LLM-Checkpoints:/work/CheckFreq/src:$PYTHONPATH
export CF_USE_THREAD=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# Optional: seed AUTO hints (CPU snapshot + thread persist)
python - <<'PY'
import json, pathlib
cfg = {"avg_iter_dur": 0.07, "chk_freq": 120, "self._chk_fn": "cpu", "self._use_thread": True}
for name in [".cache_bloom560m_1", ".cache_bloom560m_4"]:
    pathlib.Path("/work/LLM-Checkpoints/"+name).write_text(json.dumps(cfg))
print("seeded")
PY

mkdir -p /work/chk_bloom_multi

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  /work/LLM-Checkpoints/models/nlp/bloom_cf.py \
  --model bigscience/bloom-560m \
  --train-file /work/wt2_small.txt \
  --seq-len 128 \
  --batch-size 2 \
  --grad-accum-steps 4 \
  --epochs 1 \
  --workers 2 \
  --lr 2e-5 \
  --chk-prefix /work/chk_bloom_multi \
  --manual-freq 0 \
  --arch-name bloom560m | tee -a /work/chk_bloom_multi/run.log
```

---

## Troubleshooting

- **`ModuleNotFoundError: transformers`**  
  You’re in a fresh container. Re-run the **Python deps** block.

- **`/work/python: [Errno 2]`**  
  `torchrun` needs a *file path*, not `python -`. Use a script file or point to `bloom_cf.py`.

- **`CUDA error: invalid device ordinal`**  
  Ensure `CUDA_VISIBLE_DEVICES` has at least as many IDs as `--nproc_per_node`.  
  Example: for 4 ranks, `export CUDA_VISIBLE_DEVICES=0,1,2,3`.

- **`Default process group has not been initialized` during persist**  
  Keep `export CF_USE_THREAD=1` (already set above) so the async checkpoint writer uses a thread path.

- **Slow disk or large checkpoints**  
  Use MANUAL mode first (`--manual-freq 200`) to verify stability, then AUTO (`--manual-freq 0`).

---

## Add README to GitHub

**Option 1 — Web UI**
1. Download the `README.md` from this chat (link below).
2. On GitHub, open your repo → **Add file** → **Upload files** → drag `README.md` → **Commit**.

**Option 2 — Git in the container**
```bash
cd /work/LLM-Checkpoints   # or your target repo
mv /work/README.md ./README.md  # if you saved it in the mount
git checkout -b add-readme
git add README.md
git commit -m "Add README with single/multi-GPU runs"
git push origin add-readme
```
Then open a Pull Request on GitHub.

