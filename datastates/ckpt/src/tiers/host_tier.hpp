// #ifndef __DATASTATES_HOST_TIER_HPP
// #define __DATASTATES_HOST_TIER_HPP

// #include "base_tier.hpp"

// #include <atomic>
// #include <condition_variable>
// #include <deque>
// #include <filesystem>
// #include <fstream>
// #include <mutex>
// #include <thread>
// #include <vector>

// class host_tier_t : public base_tier_t {
// private:
//     char* start_ptr_ = nullptr;

//     // ---- Flush: internal MPMC queue ----
//     std::vector<std::thread> flush_threads_;
//     std::mutex flush_mtx_;
//     std::condition_variable flush_cv_;
//     std::deque<mem_region_t*> flush_deque_;
//     std::atomic<size_t> flush_outstanding_{0};

//     // wait_for_completion() helper
//     std::mutex flush_done_mtx_;
//     std::condition_variable flush_done_cv_;

//     // ---- Fetch: internal queue (single consumer thread, but multi-producer safe) ----
//     std::mutex fetch_mtx_;
//     std::condition_variable fetch_cv_;
//     std::deque<mem_region_t*> fetch_deque_;
//     std::atomic<size_t> fetch_outstanding_{0};

//     // mem_pool thread-safety가 불명확하면 보호 (안전하게 넣음)
//     std::mutex pool_mtx_;

// public:
//     host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size);
//     ~host_tier_t() override;

//     void flush(mem_region_t* m) override;
//     void fetch(mem_region_t* m) override;

//     void flush_io_();
//     void fetch_io_();

//     void wait_for_completion() override;
// };

// #endif // __DATASTATES_HOST_TIER_HPP

#ifndef __DATASTATES_HOST_TIER_HPP
#define __DATASTATES_HOST_TIER_HPP

#include "base_tier.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>

class host_tier_t : public base_tier_t {
private:
    // mem_pool_t가 char*를 요구하므로 char*로 유지
    char* start_ptr_ = nullptr;

    std::vector<std::thread> flush_threads_;

    std::mutex flush_mtx_;
    std::condition_variable flush_cv_;
    std::deque<mem_region_t*> flush_deque_;

    std::mutex fetch_mtx_;
    std::condition_variable fetch_cv_;
    std::deque<mem_region_t*> fetch_deque_;

    std::mutex flush_done_mtx_;
    std::condition_variable flush_done_cv_;

    std::atomic<size_t> flush_outstanding_{0};
    std::atomic<size_t> fetch_outstanding_{0};

    std::mutex pool_mtx_;

public:
    host_tier_t(int gpu_id, unsigned int num_threads, size_t total_size);
    ~host_tier_t() override;

    void flush(mem_region_t* src) override;
    void fetch(mem_region_t* src) override;
    void wait_for_completion() override;

    void flush_io_() override;
    void fetch_io_() override;
};

#endif // __DATASTATES_HOST_TIER_HPP