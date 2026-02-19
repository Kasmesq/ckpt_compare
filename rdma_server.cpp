#include "rdma_common.hpp"

int main(int argc, char** argv) {
  // Args: --port 18515 --size_bytes 268435456
  int port = 18515;
  size_t size = 256ULL * 1024 * 1024;
  int ib_port = 1;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
    else if (a == "--size_bytes" && i + 1 < argc) size = std::stoull(argv[++i]);
    else if (a == "--ib_port" && i + 1 < argc) ib_port = std::stoi(argv[++i]);
  }

  std::cout << "[server] size_bytes=" << size << " port=" << port << " ib_port=" << ib_port << "\n";

  int num_devs = 0;
  ibv_device** dev_list = ibv_get_device_list(&num_devs);
  if (!dev_list || num_devs == 0) die("no IB devices found");

  ibv_context* ctx = ibv_open_device(dev_list[0]);
  if (!ctx) die("ibv_open_device");

  ibv_pd* pd = ibv_alloc_pd(ctx);
  if (!pd) die("ibv_alloc_pd");

  ibv_cq* cq = ibv_create_cq(ctx, 4096, nullptr, nullptr, 0);
  if (!cq) die("ibv_create_cq");

  ibv_qp_init_attr qpia{};
  qpia.send_cq = cq;
  qpia.recv_cq = cq;
  qpia.qp_type = IBV_QPT_RC;
  qpia.cap.max_send_wr = 4096;
  qpia.cap.max_recv_wr = 16;
  qpia.cap.max_send_sge = 1;
  qpia.cap.max_recv_sge = 1;

  ibv_qp* qp = ibv_create_qp(pd, &qpia);
  if (!qp) die("ibv_create_qp");

  ibv_port_attr pattr{};
  if (ibv_query_port(ctx, ib_port, &pattr)) die("ibv_query_port");

  // Use pinned host buffer on server to reduce CPU faults.
  void* buf = nullptr;
  CHECK_CUDA(cudaHostAlloc(&buf, size, cudaHostAllocDefault));
  std::memset(buf, 0, size);

  ibv_mr* mr = ibv_reg_mr(pd, buf, size,
                          IBV_ACCESS_LOCAL_WRITE |
                          IBV_ACCESS_REMOTE_WRITE |
                          IBV_ACCESS_REMOTE_READ);
  if (!mr) die("ibv_reg_mr(server)");

  uint32_t psn = rand_psn();

  ConnInfo local{};
  local.lid = pattr.lid;
  local.qpn = qp->qp_num;
  local.psn = psn;
  local.rkey = mr->rkey;
  local.vaddr = (uint64_t)(uintptr_t)buf;

  // TCP exchange
  int lfd = tcp_listen(port);
  std::cout << "[server] waiting for TCP client on port " << port << "...\n";
  int cfd = tcp_accept(lfd);

  ConnInfo remote{};
  tcp_write_all(cfd, &local, sizeof(local));
  tcp_read_all(cfd, &remote, sizeof(remote));

  std::cout << "[server] local: lid=" << local.lid << " qpn=" << local.qpn << " psn=" << local.psn
            << " rkey=" << local.rkey << " vaddr=0x" << std::hex << local.vaddr << std::dec << "\n";
  std::cout << "[server] remote: lid=" << remote.lid << " qpn=" << remote.qpn << " psn=" << remote.psn
            << " rkey=" << remote.rkey << " vaddr=0x" << std::hex << remote.vaddr << std::dec << "\n";

  // QP state transitions
  qp_to_init(qp, ib_port);
  qp_to_rtr(qp, remote, ib_port);
  qp_to_rts(qp, psn);

  std::cout << "[server] QP is RTS. Ready for one-sided RDMA writes.\n";
  std::cout << "[server] Press Ctrl+C to exit.\n";
  while (true) ::sleep(60);
}