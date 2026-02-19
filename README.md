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
```

#Run

## Server
```bash
./rdma_server <size_bytes>
```

## Example
```bash
./rdma_server 67108864
```

## Client
```bash
./rdma_client <server_ip> <size_bytes> [host|pinned|gpu]
```

## Example
```bash
./rdma_client 192.168.100.1 67108864 host
./rdma_client 192.168.100.1 67108864 pinned
./rdma_client 192.168.100.1 67108864 gpu
```
