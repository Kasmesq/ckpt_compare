#!/usr/bin/env bash
set -euo pipefail

ENV=ds0112_torch21
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo "[DEBUG] REPO=$REPO"
echo "[DEBUG] ENV=$ENV"

conda env remove -n "$ENV" -y || true
conda create -y -n "$ENV" python=3.10 pip setuptools=80.10.2

conda run -n "$ENV" python -m pip install -U pip wheel

# torch/cu121 고정
conda run -n "$ENV" python -m pip install \
  torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# numpy / deepspeed / build deps 고정
conda run -n "$ENV" python -m pip install \
  numpy==1.26.4 \
  deepspeed==0.11.2 \
  pybind11==3.0.2

# HF stack
conda run -n "$ENV" python -m pip install \
  transformers==4.40.2 \
  datasets==2.19.2 \
  accelerate==0.30.1 \
  sentencepiece \
  safetensors \
  tokenizers

# datastates 빌드 설치
cd "$REPO"
conda run -n "$ENV" python -m pip install -v --no-build-isolation --no-deps .

# 검증
conda run -n "$ENV" python - <<'PY'
import os
import torch, deepspeed, transformers, datasets, accelerate
print("[DEBUG] torch:", torch.__version__, "cuda:", torch.version.cuda, "avail:", torch.cuda.is_available())
print("[DEBUG] deepspeed:", deepspeed.__version__)
print("[DEBUG] transformers:", transformers.__version__)
print("[DEBUG] datasets:", datasets.__version__)
print("[DEBUG] accelerate:", accelerate.__version__)
print("[DEBUG] torch lib:", os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
