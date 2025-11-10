## setup 
(One-time) Fix python alias to avoid python3.9: command not found
```bash
cd ~

# Comment out any python -> python3.9 aliases in ~/.bashrc
sed -i 's/^alias python=python3\.9/# alias python=python3.9/' ~/.bashrc
sed -i 's/^alias python3=python3\.9/# alias python3=python3.9/' ~/.bashrc

# Reload shell config
source ~/.bashrc
hash -r

# Check that python now comes from conda, not an alias
type -a python
python --version

```
Should see

```bash
python is /home/yi/miniconda3/bin/python
Python 3.12.9
```

1. Create env for PyTorch 2.1 + DeepSpeed 0.11.2 (Python 3.10)

```bash
# Create env
conda create -y -n ds0112_torch21 python=3.10

# Activate it
conda activate ds0112_torch21

# Sanity check python
python - << 'PY'
import sys
print("[DEBUG] sys.executable =", sys.executable)
print("[DEBUG] Python version =", sys.version.split()[0])
PY

```

2. Point to system CUDA 11.8 (NVCC 11.8.89)

```bash
conda activate ds0112_torch21

export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

which nvcc
nvcc --version | sed -n '1,4p'
```

Should Show:

```bash
/usr/local/cuda-11.8/bin/nvcc
nvcc: NVIDIA (R) Cuda compiler driver
...
Cuda compilation tools, release 11.8, V11.8.89
```
3. Install PyTorch 2.1.0 (CUDA 11.8 wheels)
```bash
conda activate ds0112_torch21

python -m pip install \
  torch==2.1.0+cu118 \
  torchvision==0.16.0+cu118 \
  torchaudio==2.1.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

Sanity check
```bash
conda activate ds0112_torch21

python - << 'PY'
import sys, torch
print("[DEBUG] sys.executable =", sys.executable)
print("[DEBUG] Python version =", sys.version.split()[0])
print("[DEBUG] torch.__version__  =", torch.__version__)
print("[DEBUG] torch.version.cuda =", torch.version.cuda)
print("[DEBUG] CUDA available     =", torch.cuda.is_available())
PY
```
4. Install project dependencies (DeepSpeed 0.11.2, NumPy 1.26, etc.)
```bash
conda activate ds0112_torch21

# Go to LLM-Checkpoints folder
cd ~/datastates-llm/LLM-Checkpoints

# Install pinned deps (includes deepspeed==0.11.2 and numpy==1.26.4)
python -m pip install -r dependencies.txt
```

if accidentally install Numpy 2.xx

```bash
python -m pip install "numpy<2" --force-reinstall
```

Check:

```bash
conda activate ds0112_torch21

python - << 'PY'
import torch, deepspeed
print("torch:", torch.__version__, "cuda", torch.version.cuda, "CUDA avail:", torch.cuda.is_available())
print("deepspeed:", deepspeed.__version__)
PY
```

Expected:

```bash
torch: 2.1.0+cu118 cuda 11.8 CUDA avail: True
deepspeed: 0.11.2
```

5. Build & install datastates-llm (C++/CUDA extension)
```bash
conda activate ds0112_torch21

# Make sure CUDA 11.8 is in the environment (same as above)
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Go to datastates-llm repo root
cd ~/datastates-llm
ls
# should see: datastates/, setup.py, LLM-Checkpoints/, etc.

# (Re)build C++ extension & Python package
python -m pip install -v .
```

Sanity check
```bash
conda activate ds0112_torch21
cd ~

python - << 'PY'
import sys, torch, deepspeed
import datastates
from datastates import ckpt

print("[DEBUG] sys.executable        =", sys.executable)
print("[DEBUG] Python version        =", sys.version.split()[0])
print("[DEBUG] torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
print("[DEBUG] deepspeed.__version__ =", deepspeed.__version__)
print("[DEBUG] datastates package    =", datastates)
print("[DEBUG] datastates.__file__   =", getattr(datastates, "__file__", None))
print("[DEBUG] datastates.ckpt module=", ckpt)
PY
```

6. Optional: run datastates tests
```bash
conda activate ds0112_torch21
cd ~/datastates-llm

# DataStates engine tests (C++/CUDA extension)
python datastates/tests/test_ckpt_engine.py

# LLM checkpointing tests
python datastates/tests/test_datastates_llm.py
```

7.Using in LLM-Checkpoints

```bash
conda activate ds0112_torch21
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd ~/datastates-llm/LLM-Checkpoints

# Example sanity script: just ensure imports work before real training
python - << 'PY'
import torch, deepspeed, datastates
print("torch:", torch.__version__, "cuda", torch.version.cuda, "CUDA avail:", torch.cuda.is_available())
print("deepspeed:", deepspeed.__version__)
print("datastates:", datastates)
PY

# Then run your actual deepspeed job, e.g.
# deepspeed train_xxx.py --deepspeed ds_config.json ...
```
