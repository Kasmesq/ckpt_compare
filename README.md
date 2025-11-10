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

Activate env and check Python / Torch

```bash
conda activate datastates-llm

type -a python
python --version

python - << 'PY'
import sys, torch
print("[DEBUG] sys.executable =", sys.executable)
print("[DEBUG] Python version:", sys.version.split()[0])
print("[DEBUG] torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
PY

```

Build & test datastes-llm

```bash
cd ~/datastates-llm

# (Re)build C++ extension & Python package
python -m pip install -v .

# Run DataStates engine tests (no DeepSpeed needed)
python datastates/tests/test_ckpt_engine.py
python datastates/tests/test_datastates_llm.py

```
