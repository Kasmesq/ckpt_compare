#include "engine.hpp"

datastates_llm_t::datastates_llm_t(size_t host_cache_size, int gpu_id_, int rank_)
    : gpu_id(gpu_id_), rank(rank_) {
    try {
        DBG("DataStates initing: GPU: " << gpu_id
                                        << ", host cache (MB): "
                                        << (host_cache_size >> 20));

        checkCuda(cudaSetDevice(gpu_id));
        is_active = true;

        int num_threads = 1;   // initial prototype
        size_t gpu_cache = 0;  // initial prototype

        host_tier = new host_tier_t(gpu_id, num_threads, host_cache_size);
        gpu_tier = new gpu_tier_t(gpu_id, num_threads, gpu_cache);
        gpu_tier->set_successor_tier(host_tier);

    } catch (std::exception& e) {
        FATAL("Standard exception caught in datastates init: " << e.what());
    }
}

void datastates_llm_t::ckpt_tensor(
    int version,
    const torch::Tensor &t,
    const std::uint64_t size,
    const std::uint64_t file_offset,
    std::string path
) {
    try {
        uint64_t uid = local_uid++;
        DBG("Going to checkpoint tensor of UID " << uid
                                                 << " and size " << size
                                                 << " at offset " << file_offset);

        if (t.device().is_cuda()) {
            assert((t.device().is_cuda() && t.device().index() == gpu_id) &&
                   "Tensor not on the same GPU as ckpt engine");

            mem_region_t* m = new mem_region_t(
                version,
                uid,
                static_cast<char*>(t.data_ptr()),
                size,
                file_offset,
                path,
                GPU_TIER
            );
            gpu_tier->flush(m);
            return;
        }

        mem_region_t* m = new mem_region_t(
            version,
            uid,
            static_cast<char*>(t.data_ptr()),
            size,
            file_offset,
            path,
            HOST_PINNED_TIER
        );
        host_tier->flush(m);
        return;

    } catch (std::exception& e) {
        FATAL("Exception caught in ckpt_tensor. " << e.what());
    }
}

void datastates_llm_t::restore_tensor(
    int version,
    const torch::Tensor &t,
    const std::uint64_t size,
    const std::uint64_t file_offset,
    std::string path
) {
    try {
        if (t.device().is_cuda()) {
            FATAL("Restoring GPU tensor is not yet supported");
        }

        uint64_t uid = local_uid++;
        DBG("Going to restore from " << path
                                     << " tensor of size " << size
                                     << " at file offset " << file_offset);

        mem_region_t* m = new mem_region_t(
            version,
            uid,
            static_cast<char*>(t.data_ptr()),
            size,
            file_offset,
            path,
            HOST_PINNED_TIER
        );
        host_tier->fetch(m);
        return;

    } catch (std::exception& e) {
        FATAL("Exception caught in restore_tensor. " << e.what());
    }
}

void datastates_llm_t::wait() {
    try {
        // hot-path wait:
        // GPU -> HOST staging만 drain
        if (gpu_tier != nullptr) {
            gpu_tier->wait_for_completion();
        }
    } catch (std::exception& e) {
        FATAL("Exception caught in wait(). " << e.what());
    }
}

void datastates_llm_t::wait_durable() {
    try {
        // durable wait:
        // 1) GPU -> HOST staging drain
        if (gpu_tier != nullptr) {
            gpu_tier->wait_for_completion();
        }

        // 2) HOST -> FILE flush drain
        if (host_tier != nullptr) {
            host_tier->wait_for_completion();
        }
    } catch (std::exception& e) {
        FATAL("Exception caught in wait_durable(). " << e.what());
    }
}

void datastates_llm_t::shutdown() {
    try {
        delete gpu_tier;
        gpu_tier = nullptr;

        delete host_tier;
        host_tier = nullptr;

        return;
    } catch (std::exception& e) {
        FATAL("Exception caught in shutdown(). " << e.what());
    }
}