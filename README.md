# GPUDirect RDMA Minimal Benchmark

This repository provides a minimal RC (Reliable Connection) RDMA benchmark for evaluating:

- Host memory RDMA
- Pinned host memory RDMA
- GPU Direct RDMA (via `nvidia_peermem`)

It is designed for multi-GPU training checkpoint experiments.

---

# 1. System Overview

Tested on:

- GPUs: NVIDIA Tesla V100 SXM2
- NIC: Mellanox ConnectX-6 (mlx5)
- IB Link: HDR (100 Gb/s)
- CUDA: 11.8
- Driver: nvidia_peermem enabled

## ⚠ PCIe Limitation

Although the IB link is 100 Gb/s, the NIC is operating at:



LnkSta: Speed 8GT/s (downgraded), Width x8 (downgraded)

```bash
This corresponds to PCIe Gen3 x8.

The practical RDMA ceiling in this configuration is:
```
~55–60 Gb/s

```bash
This is expected and hardware-limited.

---

# 2. Build Instructions

Activate CUDA environment:

```bash
conda activate ds0112_torch21

export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

```


```bash


```



```bash
cd gdr_rdma
rm -rf build
mkdir build
cd build
cmake ..
make -j

```


# Run

## 3.1 Start Server (sxm1)

```bash
sudo modprobe nvidia_peermem
./rdma_server --size_bytes 268435456 --port 18515 --ib_port 1

```


## 3.2 Run Client (sxm2)

## Host RDMA
```bash
taskset -c 18-20,23-24 \
./rdma_client \
  --server_ip 192.168.100.1 \
  --mode host \
  --size_bytes 268435456 \
  --iters 20

```

## GPU Direct RDMA

```bash
taskset -c 18-20,23-24 \
./rdma_client \
  --server_ip 192.168.100.1 \
  --mode gpu \
  --size_bytes 268435456 \
  --iters 20
```

## Interpreting Results

If GPU mode prints

```bash
MR registered OK
```

GPUDirect RDMA is successfully enabled

Typical host RDMA throughput:

```bash
~53~58 Gb/s
```

GPU RDMA may initially be lower if:
	•	GPU not local to NIC (topology mismatch)
	•	Very large single WR transfers
	•	No chunking/pipelining


# 5. Checkpiunt Transfer Implication

Measured throughput:
```bash
~53~58 Gb/s = 6.8 GB/s
```

A 32 GB checkpoint requires approximately:

```bash
32 GB / 6.8 GB/s ≈ 4.7 seconds (ideal case)
```

Sub-second full checkpoint transfer is not possible under PCIe Gen3 x8 constraints.

Optimization focus should be:
	•	Removing GPU→CPU staging
	•	Using GPUDirect RDMA
	•	Chunked pipelined transfers
	•	Overlapping training and transmission


# 6. Multi-GPU Training Scenario

Example:
	•	GPU0: model + optimizer
	•	GPU1/2/3: optimizer only

Recommended design:

## Aggregator GPU Strategy
	1.	Select NIC-local GPU (e.g., GPU2 or GPU3)
	2.	Use cudaMemcpyPeerAsync to aggregate states via NVLink
	3.	Perform GPUDirect RDMA from that GPU only

Advantages:
	•	Better GPU–NIC locality
	•	More stable throughput
	•	Reduced topology penalties

# 7. Summary
	•	GPUDirect RDMA is functional (MR registration succeeds)
	•	Throughput limited by PCIe Gen3 x8 (~60 Gb/s ceiling)
	•	Performance tuning requires topology awareness and chunking
	•	Ideal for checkpoint pipeline research and overlap experiments
