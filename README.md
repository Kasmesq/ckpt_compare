ssh -i ~/.ssh/Oteo.pem ubuntu@
#!/bin/bash

# 1. 시스템 패키지 업데이트 및 OS 의존성 주입 (libtinfo5 백포트)
echo "[1/8] Installing OS Dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build python3-pip python3.12-venv python3-full git wget
wget http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb
sudo dpkg -i libtinfo5_6.3-2ubuntu0.1_amd64.deb

# 2. CUDA 12.1 툴체인 구축 (PyTorch ABI 호환성 확립)
echo "[2/8] Constructing CUDA 12.1 Toolchain..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-1

# 3. 환경 변수 바인딩 (현재 세션 및 영구 세션)
echo "[3/8] Binding Environment Variables..."
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# 4. 런타임 환경 격리 (Ubuntu 24.04 PEP 668 정책 대응)
echo "[4/8] Isolating Python Runtime..."
python3 -m venv ~/llm_env
source ~/llm_env/bin/activate

# 5. 분산 훈련 프레임워크 스택 설치
echo "[5/8] Installing Framework Stack..."
pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate sentencepiece
pip install deepspeed

# 6. 타겟 브랜치 클론 및 DataStates-LLM C++ 엔진 빌드
echo "[6/8] Building DataStates-LLM Engine..."
cd ~
rm -rf ckpt_compare
git clone -b 7b-datastatesllm-a100 https://github.com/Kasmesq/ckpt_compare.git
cd ckpt_compare
pip install -e . --no-build-isolation

# 7. 아키텍처 제약사항 패치 (L40S VRAM 및 61GB Host RAM 대응)
echo "[7/8] Patching System Configurations..."
# A. 모델 가중치 텐서 타입 패치 (FP16 -> BF16)
sed -i 's/torch_dtype=torch.float16/torch_dtype=torch.bfloat16/g' datastates_train_bloom_generic_p_auto.py

# B. DeepSpeed 파서 오류 및 Host OOM 회피를 위한 JSON 재작성
# (host_cache_size 8GB 축소, 배치 산술 정규화)
cat << 'EOF' > ds_config_zero2_bf16_offload.json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 8,
  "train_batch_size": 8,
  "steps_per_print": 50,
  "wall_clock_breakdown": true,
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "reduce_scatter": true,
    "allgather_partitions": true,
    "overlap_comm": false,
    "reduce_bucket_size": 100000000,
    "allgather_bucket_size": 100000000,
    "contiguous_gradients": true,
    "round_robin_gradients": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-5,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "bf16": {
    "enabled": true
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": true,
    "cpu_checkpointing": false,
    "synchronize_checkpoint_boundary": false
  },
  "checkpoint": {
    "tag_validation_enabled": true
  },
  "datastates_ckpt": {
    "host_cache_size": 8,
    "parser_threads": 8
  }
}
EOF

# 8. 훈련 더미 데이터 생성
echo "[8/8] Generating Dummy Dataset..."
echo "This is a test sentence for BLOOM training setup on L40S." > input_data.txt
echo "End-to-End System Initialization Complete."
