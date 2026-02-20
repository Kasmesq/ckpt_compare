// #ifndef __DATASTATES_HOST_TIER_HPP
// #define __DATASTATES_HOST_TIER_HPP

// #include "base_tier.hpp"
// #include <fstream>
// #include <filesystem>

// class host_tier_t : public base_tier_t {
//     char* start_ptr_ = nullptr;
// public:
//     host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size);
//     ~host_tier_t() {
//         wait_for_completion();
//         is_active = false;
//         flush_q.set_inactive();
//         fetch_q.set_inactive();
//     };
//     void flush(mem_region_t* m);
//     void fetch(mem_region_t* m);
//     void flush_io_();
//     void fetch_io_();
//     void wait_for_completion();
// };

// #endif // __DATASTATES_HOST_TIER_HPP

#ifndef __DATASTATES_HOST_TIER_HPP
#define __DATASTATES_HOST_TIER_HPP

#include "base_tier.hpp"
#include <fstream>
#include <filesystem>
#include <vector>  // [추가] 벡터 사용
#include <thread>  // [추가] 스레드 사용
#include <mutex>   // [추가] 뮤텍스 사용

class host_tier_t : public base_tier_t {
    char* start_ptr_ = nullptr;
    
    // [변경] 단일 스레드 대신 스레드 벡터 사용
    std::vector<std::thread> flush_threads_;
    // [추가] 큐 접근 보호를 위한 뮤텍스
    std::mutex q_mtx_;

public:
    host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size);
    
    ~host_tier_t() {
        wait_for_completion();
        is_active = false;
        flush_q.set_inactive();
        fetch_q.set_inactive();

        // [변경] 모든 플러시 스레드가 종료될 때까지 대기 (Join)
        for (auto& t : flush_threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
        // fetch thread는 보통 base_tier나 멤버변수에 있으므로 기존대로 유지
        if (fetch_thread_.joinable()) fetch_thread_.join();
    };

    void flush(mem_region_t* m);
    void fetch(mem_region_t* m);
    void flush_io_();
    void fetch_io_();
    void wait_for_completion();
};

#endif // __DATASTATES_HOST_TIER_HPP