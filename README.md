1. run bash
```bash
bash
```
0) (optional) conda on the host
```bash
# install miniconda (skip if you already have conda)
cd ~
curl -fsSLo miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh -b -p "$HOME/miniconda3"
rm -f miniconda.sh
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# optional: a throwaway env
conda create -y -n tmp python=3.10 && conda activate tmp
```

conda is not required for the docker flow below; useful only if you want host tools.

1) verify nvidia driver on the host
```bash
nvidia-smi
# expect to see your GPUs and driver (e.g., 560.xx with CUDA version shown)
```
2) install nvidia-container-toolkit (gpu runtime for docker)

for ubuntu 24.04 (noble) — use the generic “stable” repo path.
```bash
# add NVIDIA repo key + list
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
 | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null <<'EOF'
deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /
EOF

sudo apt-get update

# install toolkit & friends
sudo apt-get install -y \
  nvidia-container-toolkit \
  nvidia-container-toolkit-base \
  libnvidia-container1 \
  libnvidia-container-tools

```
configure docker to use the nvidia runtime and restart:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# sanity check: docker shows 'nvidia' and default runtime is nvidia
sudo docker info | grep -i -A5 'Runtimes'
```


quick gpu test inside a plain CUDA 11.8 container:
```bash
sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
# expect a normal nvidia-smi table
```

3)workspace & code (host)
```bash
mkdir -p ~/checkfreq_env && cd ~/checkfreq_env
git clone https://github.com/msr-fiddle/CheckFreq.git
# (CoorDL not needed for the NLP test; skip)
```

4) start a PyTorch CUDA 11.8 container
```bash
sudo docker run --rm -it --gpus all --ipc=host \
  -v ~/checkfreq_env:/work \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime bash
```

5)inside the container — install python deps
```bash
pip install -U pip
pip install "transformers<4.46" datasets accelerate sentencepiece
```

6) prepare bloom_cf.py (in the github page)

7) prepare a small WikiText file
```bash
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
with open("/work/wt2_small.txt","w") as f:
    for split in ("train","validation"):
        for s in ds[split]["text"]:
            s = (s or "").strip()
            if s:
                f.write(s+"\n")
print("wrote /work/wt2_small.txt")
PY
```

8)run checkfreq(auto) + bloom-560M
```bash
# env
export HF_HOME=/work/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/work/CheckFreq:/work/CheckFreq/src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0         # change to "0,1" + --nproc_per_node=2 to use 2 GPUs
export CF_USE_THREAD=1                # thread persist => avoids CUDA OOM from forked procs

# clean AUTO cache + outputs
cd /work/CheckFreq
rm -f .cache_bloom3b_1
mkdir -p /work/chk_bloom_auto_small

# run (AUTO because --manual-freq 0)
torchrun --nproc_per_node=1 \
  /work/CheckFreq/models/nlp/bloom_cf.py \
  --model bigscience/bloom-560m \
  --train-file /work/wt2_small.txt \
  --seq-len 128 --batch-size 1 --grad-accum-steps 8 \
  --epochs 1 --workers 2 --lr 2e-5 \
  --chk-prefix /work/chk_bloom_auto_small \
  --manual-freq 0 --arch-name bloom560m \
  | tee -a /work/chk_bloom_auto_small/run.log
```
