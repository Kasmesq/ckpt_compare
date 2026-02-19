#include "rdma_common.hpp"

static void usage() {
  std::cout <<
    "Usage:\n"
    "  rdma_client --server_ip <ip> --mode host|pinned|gpu --size_bytes <N> --iters <K> [--port 18515] [--ib_port 1]\n"
    "Example:\n"
    "  rdma_client --server_ip 192.168.100.1 --mode gpu --size_bytes 268435456 --iters 200\n";
}

int main(int argc, char** argv) {
  std::string server_ip;
  std::string mode = "host";
  size_t size = 256ULL * 1024 * 1024;
  int iters = 200;
  int port = 18515;
  int ib_port = 1;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--server_ip" && i + 1 < argc) server_ip = argv[++i];
    else if (a == "--mode" && i + 1 < argc) mode = argv[++i];
    else if (a == "--size_bytes" && i + 1 < argc) size = std::stoull(argv[++i]);
    else if (a == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
    else if (a == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
    else if (a == "--ib_port" && i + 1 < argc) ib_port = std::stoi(argv[++i]);
    else if (a == "-h" || a == "--help") { usage(); return 0; }
  }

  if (server_ip.empty()) { usage(); return 1; }

  std::cout << "[client] server_ip=" << server_ip << " mode=" << mode
            << " size_bytes=" << size << " iters=" << iters
            << " port=" << port << " ib_port=" << ib_port << "\n";

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

  // Allocate buffer
  void* buf = nullptr;
  if (mode == "host") {
    buf = std::aligned_alloc(4096, ((size + 4095) / 4096) * 4096);
    if (!buf) die("aligned_alloc");
    std::memset(buf, 0xAB, size);
  } else if (mode == "pinned") {
    CHECK_CUDA(cudaHostAlloc(&buf, size, cudaHostAllocDefault));
    std::memset(buf, 0xAB, size);
  } else if (mode == "gpu") {
    CHECK_CUDA(cudaMalloc(&buf, size));
    CHECK_CUDA(cudaMemset(buf, 0xAB, size));
  } else {
    die("unknown --mode (use host|pinned|gpu)");
  }

  // Register MR (THIS is the key GPUDirect check)
  ibv_mr* mr = ibv_reg_mr(pd, buf, size,
                          IBV_ACCESS_LOCAL_WRITE |
                          IBV_ACCESS_REMOTE_WRITE |
                          IBV_ACCESS_REMOTE_READ);
  if (!mr) {
    std::cerr << "[client] ibv_reg_mr FAILED for mode=" << mode
              << " (GPUDirect likely not enabled for GPU pointers)\n";
    return 2;
  }
  std::cout << "[client] MR registered OK. lkey=" << mr->lkey << " rkey=" << mr->rkey << "\n";

  uint32_t psn = rand_psn();
  ConnInfo local{};
  local.lid = pattr.lid;
  local.qpn = qp->qp_num;
  local.psn = psn;
  local.rkey = mr->rkey;
  local.vaddr = (uint64_t)(uintptr_t)buf;

  // TCP exchange
  int sfd = tcp_connect(server_ip, port);
  ConnInfo remote{};
  tcp_read_all(sfd, &remote, sizeof(remote));
  tcp_write_all(sfd, &local, sizeof(local));

  std::cout << "[client] local: lid=" << local.lid << " qpn=" << local.qpn << " psn=" << local.psn
            << " rkey=" << local.rkey << " vaddr=0x" << std::hex << local.vaddr << std::dec << "\n";
  std::cout << "[client] remote: lid=" << remote.lid << " qpn=" << remote.qpn << " psn=" << remote.psn
            << " rkey=" << remote.rkey << " vaddr=0x" << std::hex << remote.vaddr << std::dec << "\n";

  // QP transitions
  qp_to_init(qp, ib_port);
  qp_to_rtr(qp, remote, ib_port);
  qp_to_rts(qp, psn);

  // RDMA write loop: write entire buffer each iter
  ibv_sge sge{};
  sge.addr = (uintptr_t)buf;
  sge.length = (uint32_t)size;
  sge.lkey = mr->lkey;

  ibv_send_wr wr{};
  wr.wr_id = 1;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = remote.vaddr;
  wr.wr.rdma.rkey = remote.rkey;

  std::cout << "[client] starting RDMA_WRITE bench...\n";

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; i++) {
    ibv_send_wr* bad = nullptr;
    if (ibv_post_send(qp, &wr, &bad)) die("ibv_post_send");

    ibv_wc wc{};
    int ne = 0;
    do {
      ne = ibv_poll_cq(cq, 1, &wc);
    } while (ne == 0);

    if (ne < 0) die("ibv_poll_cq");
    if (wc.status != IBV_WC_SUCCESS) {
      std::cerr << "[client] CQE error: status=" << wc.status
                << " vendor_err=" << wc.vendor_err << "\n";
      return 3;
    }

    if ((i + 1) % 10 == 0) {
      std::cout << "[client] progress " << (i + 1) << "/" << iters << "\n";
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();
  double bytes = (double)size * (double)iters;
  double GBps = bytes / sec / 1e9;
  double Gbps = GBps * 8.0;

  std::cout << "[client] DONE. time_s=" << sec
            << " total_GB=" << (bytes / 1e9)
            << " throughput_GBps=" << GBps
            << " throughput_Gbps=" << Gbps << "\n";

  return 0;
}