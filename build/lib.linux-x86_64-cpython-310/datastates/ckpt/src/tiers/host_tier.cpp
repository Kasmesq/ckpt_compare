// // #include "host_tier.hpp"

// // host_tier_t::host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size): 
// //     base_tier_t(HOST_PINNED_TIER, gpu_id, num_threads, total_size) {
// //     assert((num_threads == 1) && "[HOST_TIER] Number of flush and fetch threads should be set to 1.");
// //     checkCuda(cudaSetDevice(gpu_id_));
// //     checkCuda(cudaMallocHost(&start_ptr_, total_size));
// //     mem_pool = new mem_pool_t(start_ptr_, total_size, gpu_id);
// //     flush_thread_ = std::thread([&] { flush_io_(); });
// //     fetch_thread_ = std::thread([&] { fetch_io_(); });
// //     flush_thread_.detach();
// //     fetch_thread_.detach();
// //     DBG("Started flush and fetch threads_ on Host tier for GPU: " << gpu_id);
// // }

// // void host_tier_t::flush(mem_region_t *src) {
// //     assert((successor_tier_ != nullptr) && "[HOST_TIER] Successor tier is not set.");
// //     assert((src->curr_tier_type == HOST_PINNED_TIER) && "[HOST_TIER] Source to flush from should be a host memory type.");
// //     assert((successor_tier_->tier_type == FILE_TIER) && "[HOST_TIER] Only flush from host to file supported.");
// //     flush_q.push(src);
// // }

// // void host_tier_t::fetch(mem_region_t *src) {
// //     // assert((successor_tier_ != nullptr) && "[HOST_TIER] Successor tier is not set.");
// //     // assert((src->curr_tier_type == FILE_TIER) && "[HOST_TIER] Only fetch from file to host supported.");
// //     // assert((successor_tier_->tier_type == FILE_TIER) && "[HOST_TIER] Only fetch from file to host supported.");
// //     fetch_q.push(src);
// // }

// // void host_tier_t::wait_for_completion() {
// //     DBG("Going to invoke flush_q.wait_for_completeion()");
// //     flush_q.wait_for_completion();
// // };

// // void host_tier_t::flush_io_() {
// //     checkCuda(cudaSetDevice(gpu_id_));
// //     while(is_active) {
// //         bool res = flush_q.wait_for_item();
// //         if (res == false || is_active == false)
// //             return;
// //         mem_region_t* src = flush_q.get_front();
// //         DBG("[HOST_TIER] Flushing from host to file " << src->uid << " at file_offset " << src->file_start_offset << " at " << src->path << " tensor of size " << src->size);
// //         try {
// //             if (!std::filesystem::exists(src->path)) {
// //                 std::ofstream createFile(src->path, std::ios::binary);
// //                 createFile.close();
// //             }
// //             std::ofstream f;            
// //             f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
// //             f.open(src->path, std::ios::in | std::ios::out | std::ios::binary);
// //             f.seekp(src->file_start_offset);
// //             f.write(src->ptr, src->size);
// //             f.flush();      // This is for consistency guarantee.
// //             f.close();
// //             mem_pool->deallocate(src);
// //             flush_q.pop();
// //         } catch (const std::exception& ex) {
// //             FATAL("[HostFlush] Got exception " << ex.what());
// //         }
// //     }
// // }

// // void host_tier_t::fetch_io_() {
// //     checkCuda(cudaSetDevice(gpu_id_));
// //     while(is_active) {
// //         try {
// //             bool res = fetch_q.wait_for_item();
// //             if (res == false || is_active == false)
// //                 return;
// //             mem_region_t* src = fetch_q.get_front();
// //             DBG("Starting to fetch in background thread right now " << src->path << " from offset " << src->file_start_offset << " of size " << src->size);
// //             assert((src->ptr != nullptr) && "[HOST_TIER] Memory not allocated for fetching.");
                    
// //             std::ifstream f;            
// //             f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
// //             f.open(src->path, std::ios::in | std::ios::binary);
// //             f.seekg(src->file_start_offset);
// //             f.read(src->ptr, src->size);
// //             f.close();
// //             fetch_q.pop();
// //         } catch (const std::exception& ex) {
// //             FATAL("[HostFetch] Got exception " << ex.what());
// //         }
// //     }
// // }


// // #include "host_tier.hpp"

// // host_tier_t::host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size): 
// //     base_tier_t(HOST_PINNED_TIER, gpu_id, num_threads, total_size) {
    
// //     // [삭제] 멀티 스레드를 위해 assert 제거
// //     // assert((num_threads == 1) && "[HOST_TIER] Number of flush and fetch threads should be set to 1.");
// //     checkCuda(cudaSetDevice(gpu_id_));
// //     checkCuda(cudaMallocHost(&start_ptr_, total_size));
// //     mem_pool = new mem_pool_t(start_ptr_, total_size, gpu_id);

// //     // [변경] 4개의 병렬 플러시 스레드 생성 (NVMe 병렬 쓰기 활성화)
// //     int parallel_threads = 4; 
// //     for (int i = 0; i < parallel_threads; ++i) {
// //         flush_threads_.emplace_back(std::thread([&] { flush_io_(); }));
// //     }
    
// //     // Fetch 스레드는 1개 유지 (필요 시 늘릴 수 있음)
// //     fetch_thread_ = std::thread([&] { fetch_io_(); });
    
// //     // [중요] detach() 대신 소멸자에서 join() 하도록 변경하였으므로 여기선 제거하거나
// //     // 소멸자 로직과 맞추기 위해 joinable 상태로 둡니다. 
// //     // (기존 코드의 detach는 좀비 스레드 위험이 있어 제거하는 것이 좋습니다)
    
// //     DBG("Started " << parallel_threads << " flush threads on Host tier for GPU: " << gpu_id);
// // }

// // void host_tier_t::flush(mem_region_t *src) {
// //     assert((successor_tier_ != nullptr) && "[HOST_TIER] Successor tier is not set.");
// //     assert((src->curr_tier_type == HOST_PINNED_TIER) && "[HOST_TIER] Source to flush from should be a host memory type.");
// //     assert((successor_tier_->tier_type == FILE_TIER) && "[HOST_TIER] Only flush from host to file supported.");
// //     flush_q.push(src);
// // }

// // void host_tier_t::fetch(mem_region_t *src) {
// //     fetch_q.push(src);
// // }

// // void host_tier_t::wait_for_completion() {
// //     DBG("Going to invoke flush_q.wait_for_completion()");
// //     flush_q.wait_for_completion();
// // };

// // void host_tier_t::flush_io_() {
// //     checkCuda(cudaSetDevice(gpu_id_));
// //     while(is_active) {
// //         mem_region_t* src = nullptr;

// //         // [중요] Critical Section: 큐에서 작업 하나를 안전하게 꺼내기
// //         // 여러 스레드가 동시에 접근하므로 Mutex로 보호해야 합니다.
// //         {
// //             std::lock_guard<std::mutex> lock(q_mtx_);
            
// //             bool res = flush_q.wait_for_item();
// //             if (res == false || is_active == false)
// //                 return;
            
// //             src = flush_q.get_front();
// //             flush_q.pop(); // 꺼냈으니 큐에서 제거
// //         } // 여기서 Lock 해제 (다른 스레드가 다음 작업을 가져갈 수 있음)

// //         // [병렬 실행 구간] 디스크 쓰기 (시간이 오래 걸리는 작업)
// //         DBG("[HOST_TIER] Flushing from host to file " << src->uid << " size " << src->size);
// //         try {
// //             if (!std::filesystem::exists(src->path)) {
// //                 // 파일 생성 (경합 방지를 위해 보통 파이썬 쪽에서 관리하지만 안전장치)
// //                 std::ofstream createFile(src->path, std::ios::binary);
// //                 createFile.close();
// //             }
            
// //             std::ofstream f;            
// //             f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
// //             f.open(src->path, std::ios::in | std::ios::out | std::ios::binary);
// //             f.seekp(src->file_start_offset);
            
// //             // 파이썬에서 이미 쪼개서 보냈으므로, 여기서는 그냥 쓰면 됩니다.
// //             f.write(src->ptr, src->size);
            
// //             // f.flush(); // 성능을 위해 매번 flush 할 필요는 없음 (OS 캐시에 맡김)
// //             f.close();
            
// //             mem_pool->deallocate(src);
// //             // flush_q.pop(); // [이동됨] 위쪽 Mutex 구간 안에서 이미 pop 했습니다.
            
// //         } catch (const std::exception& ex) {
// //             FATAL("[HostFlush] Got exception " << ex.what());
// //         }
// //     }
// // }

// // void host_tier_t::fetch_io_() {
// //     checkCuda(cudaSetDevice(gpu_id_));
// //     while(is_active) {
// //         try {
// //             bool res = fetch_q.wait_for_item();
// //             if (res == false || is_active == false)
// //                 return;
// //             mem_region_t* src = fetch_q.get_front();
// //             DBG("Starting to fetch " << src->path << " offset " << src->file_start_offset);
// //             assert((src->ptr != nullptr) && "[HOST_TIER] Memory not allocated.");
                    
// //             std::ifstream f;            
// //             f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
// //             f.open(src->path, std::ios::in | std::ios::binary);
// //             f.seekg(src->file_start_offset);
// //             f.read(src->ptr, src->size);
// //             f.close();
// //             fetch_q.pop();
// //         } catch (const std::exception& ex) {
// //             FATAL("[HostFetch] Got exception " << ex.what());
// //         }
// //     }
// // }


// #include "host_tier.hpp"

// static inline void ensure_file_exists(const std::string& path) {
//     std::filesystem::path p(path);
//     auto parent = p.parent_path();
//     if (!parent.empty()) {
//         std::filesystem::create_directories(parent);
//     }

//     // open(in|out) requires existence on many platforms
//     if (!std::filesystem::exists(p)) {
//         std::ofstream createFile(p, std::ios::binary | std::ios::out);
//         createFile.close();
//     }
// }

// host_tier_t::host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size)
//     : base_tier_t(HOST_PINNED_TIER, gpu_id, num_threads, total_size) {

//     checkCuda(cudaSetDevice(gpu_id_));
//     checkCuda(cudaMallocHost(&start_ptr_, total_size));
//     mem_pool = new mem_pool_t(start_ptr_, total_size, gpu_id);

//     // 실제 flush thread 수: num_threads를 무시하지 말고 그대로 쓰거나, 고정값을 쓰려면 이유를 명시
//     int parallel_threads = (num_threads > 0) ? static_cast<int>(num_threads) : 1;
//     flush_threads_.reserve(parallel_threads);
//     for (int i = 0; i < parallel_threads; ++i) {
//         flush_threads_.emplace_back(std::thread([this] { this->flush_io_(); }));
//     }

//     // fetch thread는 base_tier_t 멤버(fetch_thread_) 사용
//     fetch_thread_ = std::thread([this] { this->fetch_io_(); });

//     DBG("Started " << parallel_threads << " flush threads on Host tier for GPU: " << gpu_id);
// }

// host_tier_t::~host_tier_t() {
//     // 1) 먼저 남아있는 작업을 모두 처리하도록 기다림
//     wait_for_completion();

//     // 2) thread 종료 신호
//     is_active = false;

//     // 3) 잠든 thread 깨우기
//     flush_cv_.notify_all();
//     fetch_cv_.notify_all();

//     // 4) join
//     for (auto& t : flush_threads_) {
//         if (t.joinable()) t.join();
//     }
//     if (fetch_thread_.joinable()) fetch_thread_.join();

//     // (선택) base_tier 내부 큐를 쓰던 코드가 남아있다면 inactive 처리
//     // flush_q.set_inactive();
//     // fetch_q.set_inactive();
// }

// void host_tier_t::flush(mem_region_t* src) {
//     assert((successor_tier_ != nullptr) && "[HOST_TIER] Successor tier is not set.");
//     assert((src->curr_tier_type == HOST_PINNED_TIER) && "[HOST_TIER] Source to flush from should be HOST tier.");
//     assert((successor_tier_->tier_type == FILE_TIER) && "[HOST_TIER] Only flush host->file supported.");

//     // internal queue enqueue
//     {
//         std::lock_guard<std::mutex> lk(flush_mtx_);
//         flush_deque_.push_back(src);
//         flush_outstanding_.fetch_add(1, std::memory_order_relaxed);

//         // Debug: 큐 쌓이는지 확인 (병목 추적)
//         DBG("[HOST_TIER] enqueue uid=" << src->uid
//             << " size=" << src->size
//             << " qsize=" << flush_deque_.size()
//             << " outstanding=" << flush_outstanding_.load());
//     }
//     flush_cv_.notify_one();
// }

// void host_tier_t::fetch(mem_region_t* src) {
//     // internal queue enqueue
//     {
//         std::lock_guard<std::mutex> lk(fetch_mtx_);
//         fetch_deque_.push_back(src);
//         fetch_outstanding_.fetch_add(1, std::memory_order_relaxed);

//         DBG("[HOST_TIER] fetch enqueue uid=" << src->uid
//             << " size=" << src->size
//             << " qsize=" << fetch_deque_.size()
//             << " outstanding=" << fetch_outstanding_.load());
//     }
//     fetch_cv_.notify_one();
// }

// void host_tier_t::wait_for_completion() {
//     // flush outstanding이 0이 될 때까지 기다림
//     {
//         std::unique_lock<std::mutex> lk(flush_done_mtx_);
//         flush_done_cv_.wait(lk, [&] {
//             return flush_outstanding_.load(std::memory_order_relaxed) == 0;
//         });
//     }

//     // fetch도 쓰는 경로면 같이 기다리기 (안 쓰면 없어도 됨)
//     // fetch는 보통 restore 경로라 학습 중에는 0일 가능성이 높음
//     while (fetch_outstanding_.load(std::memory_order_relaxed) != 0) {
//         std::this_thread::sleep_for(std::chrono::milliseconds(1));
//     }

//     DBG("[HOST_TIER] wait_for_completion DONE");
// }

// void host_tier_t::flush_io_() {
//     checkCuda(cudaSetDevice(gpu_id_));

//     while (true) {
//         mem_region_t* src = nullptr;

//         // ✅ 핵심: wait는 cv로, 락은 여기서만 잡고 바로 풀기
//         {
//             std::unique_lock<std::mutex> lk(flush_mtx_);
//             flush_cv_.wait(lk, [&] {
//                 return (!is_active) || (!flush_deque_.empty());
//             });

//             // 종료 조건
//             if (!is_active && flush_deque_.empty()) {
//                 return;
//             }

//             // pop
//             src = flush_deque_.front();
//             flush_deque_.pop_front();

//             DBG("[HOST_TIER] dequeue uid=" << src->uid
//                 << " qsize=" << flush_deque_.size()
//                 << " tid=" << std::this_thread::get_id());
//         } // ✅ 락 해제 후 I/O

//         // ---- 병렬 I/O 구간 ----
//         // try {
//         //     // 파일 생성/존재 보장
//         //     ensure_file_exists(src->path);

//         //     std::ofstream f;
//         //     f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
//         //     f.open(src->path, std::ios::in | std::ios::out | std::ios::binary);
//         //     f.seekp(src->file_start_offset);
//         //     f.write(src->ptr, src->size);
//         //     f.close();

//         //     // mem_pool thread-safety 불명확하니 보호
//         //     {
//         //         std::lock_guard<std::mutex> plk(pool_mtx_);
//         //         mem_pool->deallocate(src);
//         //     }

//         // } catch (const std::exception& ex) {
//         //     // 실패해도 outstanding 감소는 해줘야 wait가 영원히 안 걸림
//         //     flush_outstanding_.fetch_sub(1, std::memory_order_relaxed);
//         //     flush_done_cv_.notify_all();
//         //     FATAL("[HostFlush] Got exception " << ex.what());
//         // }
//         try {
//             ensure_file_exists(src->path);

//             DBG("[HOST_TIER] flush start uid=" << src->uid
//                 << " path=" << src->path
//                 << " off=" << src->file_start_offset
//                 << " size=" << src->size
//                 << " tid=" << std::this_thread::get_id());

//             std::fstream f(src->path, std::ios::in | std::ios::out | std::ios::binary);
//             if (!f.is_open()) {
//                 throw std::ios_base::failure("open failed");
//             }

//             f.exceptions(std::ios::failbit | std::ios::badbit);
//             f.seekp(static_cast<std::streamoff>(src->file_start_offset), std::ios::beg);
//             f.write(src->ptr, static_cast<std::streamsize>(src->size));
//             f.flush();
//             f.close();

//             {
//                 std::lock_guard<std::mutex> plk(pool_mtx_);
//                 mem_pool->deallocate(src);
//             }

//             DBG("[HOST_TIER] flush done uid=" << src->uid
//                 << " path=" << src->path
//                 << " off=" << src->file_start_offset
//                 << " size=" << src->size
//                 << " tid=" << std::this_thread::get_id());

//         } catch (const std::exception& ex) {
//             flush_outstanding_.fetch_sub(1, std::memory_order_relaxed);
//             flush_done_cv_.notify_all();
//             FATAL("[HostFlush] path=" << src->path
//                   << " off=" << src->file_start_offset
//                   << " size=" << src->size
//                   << " gpu=" << gpu_id_
//                   << " tid=" << std::this_thread::get_id()
//                   << " ex=" << ex.what());
//         }
//         // 완료 처리
//         flush_outstanding_.fetch_sub(1, std::memory_order_relaxed);
//         flush_done_cv_.notify_all();
//     }
// }

// void host_tier_t::fetch_io_() {
//     checkCuda(cudaSetDevice(gpu_id_));

//     while (true) {
//         mem_region_t* src = nullptr;

//         {
//             std::unique_lock<std::mutex> lk(fetch_mtx_);
//             fetch_cv_.wait(lk, [&] {
//                 return (!is_active) || (!fetch_deque_.empty());
//             });

//             if (!is_active && fetch_deque_.empty()) {
//                 return;
//             }

//             src = fetch_deque_.front();
//             fetch_deque_.pop_front();

//             DBG("[HOST_TIER] fetch dequeue uid=" << src->uid
//                 << " qsize=" << fetch_deque_.size());
//         }

//         try {
//             assert((src->ptr != nullptr) && "[HOST_TIER] fetch: Memory not allocated.");

//             std::ifstream f;
//             f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
//             f.open(src->path, std::ios::in | std::ios::binary);
//             f.seekg(src->file_start_offset);
//             f.read(src->ptr, src->size);
//             f.close();

//         } catch (const std::exception& ex) {
//             fetch_outstanding_.fetch_sub(1, std::memory_order_relaxed);
//             FATAL("[HostFetch] Got exception " << ex.what());
//         }

//         fetch_outstanding_.fetch_sub(1, std::memory_order_relaxed);
//     }
// }
#include "host_tier.hpp"

#include <chrono>
#include <iostream>

static inline void ensure_file_exists(const std::string& path) {
    std::filesystem::path p(path);
    auto parent = p.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    // open(in|out) requires existence on many platforms
    if (!std::filesystem::exists(p)) {
        std::ofstream createFile(p, std::ios::binary | std::ios::out);
        createFile.close();
    }
}

host_tier_t::host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size)
    : base_tier_t(HOST_PINNED_TIER, gpu_id, num_threads, total_size) {

    checkCuda(cudaSetDevice(gpu_id_));

    // start_ptr_는 char* 이므로 void**로 캐스팅해서 cudaMallocHost 호출
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&start_ptr_), total_size));

    mem_pool = new mem_pool_t(start_ptr_, total_size, gpu_id);

    int parallel_threads = (num_threads > 0) ? static_cast<int>(num_threads) : 1;

    flush_threads_.reserve(parallel_threads);
    for (int i = 0; i < parallel_threads; ++i) {
        flush_threads_.emplace_back([this]() { this->flush_io_(); });
    }

    // base_tier_t의 fetch_thread_ 재사용
    fetch_thread_ = std::thread([this]() { this->fetch_io_(); });

    std::cerr
        << "[HOST_TIER] constructor gpu=" << gpu_id_
        << " flush_threads=" << parallel_threads
        << " total_size=" << total_size_
        << std::endl;
}

host_tier_t::~host_tier_t() {
    std::cerr << "[HOST_TIER] destructor BEGIN gpu=" << gpu_id_ << std::endl;

    // 1) outstanding 작업이 있으면 먼저 끝날 때까지 대기
    wait_for_completion();

    // 2) worker 종료 신호
    is_active.store(false, std::memory_order_release);

    // legacy queue 깨우기
    try { flush_q.set_inactive(); } catch (...) {}
    try { fetch_q.set_inactive(); } catch (...) {}

    // 3) cv wait 중인 worker 깨우기
    flush_cv_.notify_all();
    fetch_cv_.notify_all();

    // 4) flush worker join
    for (auto& t : flush_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }

    // 5) fetch worker join
    if (fetch_thread_.joinable()) {
        fetch_thread_.join();
    }

    // 6) 리소스 해제
    if (mem_pool != nullptr) {
        delete mem_pool;
        mem_pool = nullptr;
    }

    if (start_ptr_ != nullptr) {
        cudaError_t err = cudaFreeHost(start_ptr_);
        if (err != cudaSuccess) {
            std::cerr
                << "[HOST_TIER] cudaFreeHost failed gpu=" << gpu_id_
                << " err=" << cudaGetErrorString(err)
                << std::endl;
        }
        start_ptr_ = nullptr;
    }

    std::cerr << "[HOST_TIER] destructor END gpu=" << gpu_id_ << std::endl;
}

void host_tier_t::flush(mem_region_t* src) {
    assert((successor_tier_ != nullptr) && "[HOST_TIER] Successor tier is not set.");
    assert((src->curr_tier_type == HOST_PINNED_TIER) && "[HOST_TIER] Source to flush from should be HOST tier.");
    assert((successor_tier_->tier_type_ == FILE_TIER) && "[HOST_TIER] Only flush host->file supported.");

    {
        std::lock_guard<std::mutex> lk(flush_mtx_);
        flush_deque_.push_back(src);
        flush_outstanding_.fetch_add(1, std::memory_order_relaxed);

        std::cerr
            << "[HOST_TIER] enqueue uid=" << src->uid
            << " size=" << src->size
            << " qsize=" << flush_deque_.size()
            << " outstanding=" << flush_outstanding_.load(std::memory_order_relaxed)
            << std::endl;
    }

    flush_cv_.notify_one();
}

void host_tier_t::fetch(mem_region_t* src) {
    {
        std::lock_guard<std::mutex> lk(fetch_mtx_);
        fetch_deque_.push_back(src);
        fetch_outstanding_.fetch_add(1, std::memory_order_relaxed);

        std::cerr
            << "[HOST_TIER] fetch enqueue uid=" << src->uid
            << " size=" << src->size
            << " qsize=" << fetch_deque_.size()
            << " outstanding=" << fetch_outstanding_.load(std::memory_order_relaxed)
            << std::endl;
    }

    fetch_cv_.notify_one();
}

void host_tier_t::wait_for_completion() {
    {
        std::unique_lock<std::mutex> lk(flush_done_mtx_);
        flush_done_cv_.wait(lk, [&]() {
            return flush_outstanding_.load(std::memory_order_relaxed) == 0;
        });
    }

    while (fetch_outstanding_.load(std::memory_order_relaxed) != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::cerr << "[HOST_TIER] wait_for_completion DONE gpu=" << gpu_id_ << std::endl;
}

void host_tier_t::flush_io_() {
    checkCuda(cudaSetDevice(gpu_id_));

    while (true) {
        mem_region_t* src = nullptr;

        {
            std::unique_lock<std::mutex> lk(flush_mtx_);
            flush_cv_.wait(lk, [&]() {
                return (!is_active.load(std::memory_order_relaxed)) || (!flush_deque_.empty());
            });

            if (!is_active.load(std::memory_order_relaxed) && flush_deque_.empty()) {
                return;
            }

            src = flush_deque_.front();
            flush_deque_.pop_front();

            std::cerr
                << "[HOST_TIER] dequeue uid=" << src->uid
                << " qsize=" << flush_deque_.size()
                << " tid=" << std::this_thread::get_id()
                << std::endl;
        }

        try {
            ensure_file_exists(src->path);

            std::cerr
                << "[HOST_TIER] flush start uid=" << src->uid
                << " path=" << src->path
                << " off=" << src->file_start_offset
                << " size=" << src->size
                << " tid=" << std::this_thread::get_id()
                << std::endl;

            std::fstream f(src->path, std::ios::in | std::ios::out | std::ios::binary);
            if (!f.is_open()) {
                throw std::ios_base::failure("open failed");
            }

            f.exceptions(std::ios::failbit | std::ios::badbit);
            f.seekp(static_cast<std::streamoff>(src->file_start_offset), std::ios::beg);
            f.write(src->ptr, static_cast<std::streamsize>(src->size));
            f.flush();
            f.close();

            {
                std::lock_guard<std::mutex> plk(pool_mtx_);
                mem_pool->deallocate(src);
            }

            std::cerr
                << "[HOST_TIER] flush done uid=" << src->uid
                << " path=" << src->path
                << " off=" << src->file_start_offset
                << " size=" << src->size
                << " tid=" << std::this_thread::get_id()
                << std::endl;

        } catch (const std::exception& ex) {
            flush_outstanding_.fetch_sub(1, std::memory_order_relaxed);
            flush_done_cv_.notify_all();

            FATAL("[HostFlush] path=" << src->path
                  << " off=" << src->file_start_offset
                  << " size=" << src->size
                  << " gpu=" << gpu_id_
                  << " tid=" << std::this_thread::get_id()
                  << " ex=" << ex.what());
        }

        flush_outstanding_.fetch_sub(1, std::memory_order_relaxed);
        flush_done_cv_.notify_all();
    }
}

void host_tier_t::fetch_io_() {
    checkCuda(cudaSetDevice(gpu_id_));

    while (true) {
        mem_region_t* src = nullptr;

        {
            std::unique_lock<std::mutex> lk(fetch_mtx_);
            fetch_cv_.wait(lk, [&]() {
                return (!is_active.load(std::memory_order_relaxed)) || (!fetch_deque_.empty());
            });

            if (!is_active.load(std::memory_order_relaxed) && fetch_deque_.empty()) {
                return;
            }

            src = fetch_deque_.front();
            fetch_deque_.pop_front();

            std::cerr
                << "[HOST_TIER] fetch dequeue uid=" << src->uid
                << " qsize=" << fetch_deque_.size()
                << " tid=" << std::this_thread::get_id()
                << std::endl;
        }

        try {
            assert((src->ptr != nullptr) && "[HOST_TIER] fetch: Memory not allocated.");

            std::ifstream f;
            f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            f.open(src->path, std::ios::in | std::ios::binary);
            f.seekg(static_cast<std::streamoff>(src->file_start_offset), std::ios::beg);
            f.read(src->ptr, static_cast<std::streamsize>(src->size));
            f.close();

        } catch (const std::exception& ex) {
            fetch_outstanding_.fetch_sub(1, std::memory_order_relaxed);
            FATAL("[HostFetch] path=" << src->path
                  << " off=" << src->file_start_offset
                  << " size=" << src->size
                  << " gpu=" << gpu_id_
                  << " tid=" << std::this_thread::get_id()
                  << " ex=" << ex.what());
        }

        fetch_outstanding_.fetch_sub(1, std::memory_order_relaxed);
    }
}