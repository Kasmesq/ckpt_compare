Conda
```bash
conda create -y -n new_deepspeed python=3.10

conda activate new_deepspeed

# CUDA 11.8 경로 (sxm1에서 쓰던 거 그대로)
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```
Torch + CUDA
```bash
pip install \
  "torch==2.5.1" \
  "torchvision==0.20.1" \
  "torchaudio==2.5.1" \
  --index-url https://download.pytorch.org/whl/cu124
```

Dependencies
```bash
pip install \
  "deepspeed==0.17.5" \
  "transformers==4.56.1" \
  "tokenizers==0.22.0" \
  "datasets==4.0.0" \
  "hf-xet==1.1.10" \
  "einops==0.8.1" \
  "pandas==2.3.2" \
  "pyarrow==21.0.0" \
  "safetensors==0.6.2" \
  "msgpack==1.1.1" \
  "ninja==1.13.0" \
  "psutil==7.0.0" \
  "tqdm==4.67.1" \
  "regex==2025.9.1"
```
Version check
```bash
python - << 'PY'
import deepspeed, inspect, os
print("[DEBUG] deepspeed:", deepspeed.__version__)

import deepspeed.checkpoint as ds_ckpt
print("[DEBUG] deepspeed.checkpoint module:", ds_ckpt)

from deepspeed.checkpoint.decoupled_checkpoint_engine import DecoupledCheckpointEngine
print("[DEBUG] DecoupledCheckpointEngine:", DecoupledCheckpointEngine)
PY
```
