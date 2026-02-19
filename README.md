# GPUDirect RDMA Minimal Benchmark

This repository provides a minimal RDMA RC benchmark for testing:

- Host memory RDMA
- Pinned host memory RDMA
- GPU Direct RDMA

## Requirements

- Mellanox ConnectX (mlx5)
- NVIDIA GPU (V100/A100 etc.)
- `nvidia_peermem` loaded
- rdma-core installed
- CUDA installed

## Build

```bash
mkdir build
cd build
cmake ..
make -j
