// #include "host_tier.hpp"

// host_tier_t::host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size): 
//     base_tier_t(HOST_PINNED_TIER, gpu_id, num_threads, total_size) {
//     assert((num_threads == 1) && "[HOST_TIER] Number of flush and fetch threads should be set to 1.");
//     checkCuda(cudaSetDevice(gpu_id_));
//     checkCuda(cudaMallocHost(&start_ptr_, total_size));
//     mem_pool = new mem_pool_t(start_ptr_, total_size, gpu_id);
//     flush_thread_ = std::thread([&] { flush_io_(); });
//     fetch_thread_ = std::thread([&] { fetch_io_(); });
//     flush_thread_.detach();
//     fetch_thread_.detach();
//     DBG("Started flush and fetch threads_ on Host tier for GPU: " << gpu_id);
// }

// void host_tier_t::flush(mem_region_t *src) {
//     assert((successor_tier_ != nullptr) && "[HOST_TIER] Successor tier is not set.");
//     assert((src->curr_tier_type == HOST_PINNED_TIER) && "[HOST_TIER] Source to flush from should be a host memory type.");
//     assert((successor_tier_->tier_type == FILE_TIER) && "[HOST_TIER] Only flush from host to file supported.");
//     flush_q.push(src);
// }

// void host_tier_t::fetch(mem_region_t *src) {
//     // assert((successor_tier_ != nullptr) && "[HOST_TIER] Successor tier is not set.");
//     // assert((src->curr_tier_type == FILE_TIER) && "[HOST_TIER] Only fetch from file to host supported.");
//     // assert((successor_tier_->tier_type == FILE_TIER) && "[HOST_TIER] Only fetch from file to host supported.");
//     fetch_q.push(src);
// }

// void host_tier_t::wait_for_completion() {
//     DBG("Going to invoke flush_q.wait_for_completeion()");
//     flush_q.wait_for_completion();
// };

// void host_tier_t::flush_io_() {
//     checkCuda(cudaSetDevice(gpu_id_));
//     while(is_active) {
//         bool res = flush_q.wait_for_item();
//         if (res == false || is_active == false)
//             return;
//         mem_region_t* src = flush_q.get_front();
//         DBG("[HOST_TIER] Flushing from host to file " << src->uid << " at file_offset " << src->file_start_offset << " at " << src->path << " tensor of size " << src->size);
//         try {
//             if (!std::filesystem::exists(src->path)) {
//                 std::ofstream createFile(src->path, std::ios::binary);
//                 createFile.close();
//             }
//             std::ofstream f;            
//             f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
//             f.open(src->path, std::ios::in | std::ios::out | std::ios::binary);
//             f.seekp(src->file_start_offset);
//             f.write(src->ptr, src->size);
//             f.flush();      // This is for consistency guarantee.
//             f.close();
//             mem_pool->deallocate(src);
//             flush_q.pop();
//         } catch (const std::exception& ex) {
//             FATAL("[HostFlush] Got exception " << ex.what());
//         }
//     }
// }

// void host_tier_t::fetch_io_() {
//     checkCuda(cudaSetDevice(gpu_id_));
//     while(is_active) {
//         try {
//             bool res = fetch_q.wait_for_item();
//             if (res == false || is_active == false)
//                 return;
//             mem_region_t* src = fetch_q.get_front();
//             DBG("Starting to fetch in background thread right now " << src->path << " from offset " << src->file_start_offset << " of size " << src->size);
//             assert((src->ptr != nullptr) && "[HOST_TIER] Memory not allocated for fetching.");
                    
//             std::ifstream f;            
//             f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
//             f.open(src->path, std::ios::in | std::ios::binary);
//             f.seekg(src->file_start_offset);
//             f.read(src->ptr, src->size);
//             f.close();
//             fetch_q.pop();
//         } catch (const std::exception& ex) {
//             FATAL("[HostFetch] Got exception " << ex.what());
//         }
//     }
// }


#include "host_tier.hpp"

host_tier_t::host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size): 
    base_tier_t(HOST_PINNED_TIER, gpu_id, num_threads, total_size) {
    
    // [삭제] 멀티 스레드를 위해 assert 제거
    // assert((num_threads == 1) && "[HOST_TIER] Number of flush and fetch threads should be set to 1.");
    
    checkCuda(cudaSetDevice(gpu_id_));
    checkCuda(cudaMallocHost(&start_ptr_, total_size));
    mem_pool = new mem_pool_t(start_ptr_, total_size, gpu_id);

    // [변경] 4개의 병렬 플러시 스레드 생성 (NVMe 병렬 쓰기 활성화)
    int parallel_threads = 4; 
    for (int i = 0; i < parallel_threads; ++i) {
        flush_threads_.emplace_back(std::thread([&] { flush_io_(); }));
    }
    
    // Fetch 스레드는 1개 유지 (필요 시 늘릴 수 있음)
    fetch_thread_ = std::thread([&] { fetch_io_(); });
    
    // [중요] detach() 대신 소멸자에서 join() 하도록 변경하였으므로 여기선 제거하거나
    // 소멸자 로직과 맞추기 위해 joinable 상태로 둡니다. 
    // (기존 코드의 detach는 좀비 스레드 위험이 있어 제거하는 것이 좋습니다)
    
    DBG("Started " << parallel_threads << " flush threads on Host tier for GPU: " << gpu_id);
}

void host_tier_t::flush(mem_region_t *src) {
    assert((successor_tier_ != nullptr) && "[HOST_TIER] Successor tier is not set.");
    assert((src->curr_tier_type == HOST_PINNED_TIER) && "[HOST_TIER] Source to flush from should be a host memory type.");
    assert((successor_tier_->tier_type == FILE_TIER) && "[HOST_TIER] Only flush from host to file supported.");
    flush_q.push(src);
}

void host_tier_t::fetch(mem_region_t *src) {
    fetch_q.push(src);
}

void host_tier_t::wait_for_completion() {
    DBG("Going to invoke flush_q.wait_for_completion()");
    flush_q.wait_for_completion();
};

void host_tier_t::flush_io_() {
    checkCuda(cudaSetDevice(gpu_id_));
    while(is_active) {
        mem_region_t* src = nullptr;

        // [중요] Critical Section: 큐에서 작업 하나를 안전하게 꺼내기
        // 여러 스레드가 동시에 접근하므로 Mutex로 보호해야 합니다.
        {
            std::lock_guard<std::mutex> lock(q_mtx_);
            
            bool res = flush_q.wait_for_item();
            if (res == false || is_active == false)
                return;
            
            src = flush_q.get_front();
            flush_q.pop(); // 꺼냈으니 큐에서 제거
        } // 여기서 Lock 해제 (다른 스레드가 다음 작업을 가져갈 수 있음)

        // [병렬 실행 구간] 디스크 쓰기 (시간이 오래 걸리는 작업)
        DBG("[HOST_TIER] Flushing from host to file " << src->uid << " size " << src->size);
        try {
            if (!std::filesystem::exists(src->path)) {
                // 파일 생성 (경합 방지를 위해 보통 파이썬 쪽에서 관리하지만 안전장치)
                std::ofstream createFile(src->path, std::ios::binary);
                createFile.close();
            }
            
            std::ofstream f;            
            f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
            f.open(src->path, std::ios::in | std::ios::out | std::ios::binary);
            f.seekp(src->file_start_offset);
            
            // 파이썬에서 이미 쪼개서 보냈으므로, 여기서는 그냥 쓰면 됩니다.
            f.write(src->ptr, src->size);
            
            // f.flush(); // 성능을 위해 매번 flush 할 필요는 없음 (OS 캐시에 맡김)
            f.close();
            
            mem_pool->deallocate(src);
            // flush_q.pop(); // [이동됨] 위쪽 Mutex 구간 안에서 이미 pop 했습니다.
            
        } catch (const std::exception& ex) {
            FATAL("[HostFlush] Got exception " << ex.what());
        }
    }
}

void host_tier_t::fetch_io_() {
    checkCuda(cudaSetDevice(gpu_id_));
    while(is_active) {
        try {
            bool res = fetch_q.wait_for_item();
            if (res == false || is_active == false)
                return;
            mem_region_t* src = fetch_q.get_front();
            DBG("Starting to fetch " << src->path << " offset " << src->file_start_offset);
            assert((src->ptr != nullptr) && "[HOST_TIER] Memory not allocated.");
                    
            std::ifstream f;            
            f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            f.open(src->path, std::ios::in | std::ios::binary);
            f.seekg(src->file_start_offset);
            f.read(src->ptr, src->size);
            f.close();
            fetch_q.pop();
        } catch (const std::exception& ex) {
            FATAL("[HostFetch] Got exception " << ex.what());
        }
    }
}