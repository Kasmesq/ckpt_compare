#pragma once
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    std::cerr << "[CUDA] " << cudaGetErrorString(_e) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    std::exit(1); \
  } \
} while(0)

inline void die(const std::string& msg) {
  std::cerr << "[FATAL] " << msg << " errno=" << errno << " (" << std::strerror(errno) << ")\n";
  std::exit(1);
}

inline uint32_t rand_psn() {
  static std::mt19937 rng{std::random_device{}()};
  return (uint32_t)(rng() & 0xffffff);
}

struct ConnInfo {
  uint16_t lid;
  uint32_t qpn;
  uint32_t psn;
  uint32_t rkey;
  uint64_t vaddr;
};

inline int tcp_listen(int port) {
  int s = ::socket(AF_INET, SOCK_STREAM, 0);
  if (s < 0) die("socket()");
  int one = 1;
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = INADDR_ANY;
  if (bind(s, (sockaddr*)&addr, sizeof(addr)) < 0) die("bind()");
  if (listen(s, 1) < 0) die("listen()");
  return s;
}

inline int tcp_accept(int listen_fd) {
  int c = accept(listen_fd, nullptr, nullptr);
  if (c < 0) die("accept()");
  return c;
}

inline int tcp_connect(const std::string& ip, int port) {
  int s = ::socket(AF_INET, SOCK_STREAM, 0);
  if (s < 0) die("socket()");
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) != 1) die("inet_pton()");
  if (connect(s, (sockaddr*)&addr, sizeof(addr)) < 0) die("connect()");
  return s;
}

inline void tcp_write_all(int fd, const void* p, size_t n) {
  const uint8_t* b = (const uint8_t*)p;
  size_t off = 0;
  while (off < n) {
    ssize_t r = ::write(fd, b + off, n - off);
    if (r <= 0) die("write()");
    off += (size_t)r;
  }
}

inline void tcp_read_all(int fd, void* p, size_t n) {
  uint8_t* b = (uint8_t*)p;
  size_t off = 0;
  while (off < n) {
    ssize_t r = ::read(fd, b + off, n - off);
    if (r <= 0) die("read()");
    off += (size_t)r;
  }
}

// QP state transitions for RC
inline void qp_to_init(ibv_qp* qp, int ib_port) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = ib_port;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE;

  int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  if (ibv_modify_qp(qp, &attr, flags)) die("ibv_modify_qp(INIT) failed");
}

inline void qp_to_rtr(ibv_qp* qp, const ConnInfo& remote, int ib_port) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_4096;
  attr.dest_qp_num = remote.qpn;
  attr.rq_psn = remote.psn;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;

  attr.ah_attr.is_global = 0;
  attr.ah_attr.dlid = remote.lid;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = ib_port;

  int flags =
      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
      IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
      IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

  if (ibv_modify_qp(qp, &attr, flags)) die("ibv_modify_qp(RTR) failed");
}

inline void qp_to_rts(ibv_qp* qp, uint32_t local_psn) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = local_psn;
  attr.max_rd_atomic = 1;

  int flags =
      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
      IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

  if (ibv_modify_qp(qp, &attr, flags)) die("ibv_modify_qp(RTS) failed");
}