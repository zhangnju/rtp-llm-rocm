#include "src/fastertransformer/rocm/allocator_rocm.h"
#include <mutex>

namespace fastertransformer {
#define check_cuda_error(val) {}
void* IRocmAllocator::reMalloc(void* ptr, size_t size, const bool is_set_zero) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    size              = ((size + 31) / 32) * 32;  // make the buffer align with 32 bytes
    void* void_ptr    = (void*)ptr;
    void* ptr_address = void_ptr;
    if (isExist(ptr_address)) {
        ReallocType realloc_type = isReMalloc(ptr_address, size);
        if (realloc_type == ReallocType::INCREASE) {
            FT_LOG_DEBUG("ReMalloc the buffer %p since it is too small.", void_ptr);
            free((void**)(&void_ptr));
            return malloc(size, is_set_zero);
        } else if (realloc_type == ReallocType::DECREASE) {
            FT_LOG_DEBUG("ReMalloc the buffer %p to release unused memory to memory pools.", void_ptr);
            free((void**)(&void_ptr));
            return malloc(size, is_set_zero);
        } else {
            FT_LOG_DEBUG("Reuse original buffer %p with size %d and do nothing for reMalloc.", void_ptr, size);
            if (is_set_zero) {
                memSet(void_ptr, 0, size);
            }
            return void_ptr;
        }
    } else {
        FT_LOG_DEBUG("Cannot find buffer %p, mallocing new one.", void_ptr);
        return malloc(size, is_set_zero);
    }
}


void IRocmAllocator::memSet(void* ptr, const int val, const size_t size) const {
    hipMemsetAsync(ptr, val, size, stream_);
}

// cuda allocator

Allocator<AllocatorType::ROCM>::Allocator(int device_id): PurePointerRocmAllocator(device_id) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int device_count = 1;
    hipGetDeviceCount(&device_count);
    hipMemPool_t mempool;
    hipDeviceGetDefaultMemPool(&mempool, device_id);
    hipMemAccessDesc desc                  = {};
    int               peer_access_available = 0;
    for (int i = 0; i < device_count; i++) {
        if (i == device_id) {
            continue;
        }
        check_cuda_error(hipDeviceCanAccessPeer(&peer_access_available, device_id, i));
        if (!peer_access_available) {
            FT_LOG_WARNING("Device " + std::to_string(device_id) + " peer access Device " + std::to_string(i)
                            + " is not available.");
            continue;
        }
        desc.location.type = hipMemLocationTypeDevice;
        desc.location.id   = i;
        desc.flags         = hipMemAccessFlagsProtReadWrite;
        hipMemPoolSetAccess(mempool, &desc, 1);
    }
    // set memory pool threshold to avoid shrinking the pool
    uint64_t setVal = UINT64_MAX;
    hipMemPoolSetAttribute(mempool, hipMemPoolAttrReleaseThreshold, &setVal);
}

Allocator<AllocatorType::ROCM>::~Allocator() {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    while (!pointer_mapping_->empty()) {
        free((void**)(&pointer_mapping_->begin()->first));
    }
}

hipError_t getSetDevice(int i_device, int* o_device) {
    int         current_dev_id = 0;
    hipError_t err            = hipSuccess;

    if (o_device != NULL) {
        err = hipGetDevice(&current_dev_id);
        if (err != hipSuccess) {
            return err;
        }
        if (current_dev_id == i_device) {
            *o_device = i_device;
        } else {
            err = hipSetDevice(i_device);
            if (err != hipSuccess) {
                return err;
            }
            *o_device = current_dev_id;
        }
    } else {
        err = hipSetDevice(i_device);
        if (err != hipSuccess) {
            return err;
        }
    }

    return hipSuccess;
}

void* Allocator<AllocatorType::ROCM>::malloc(size_t size, const bool is_set_zero) {
    if (size == 0) {
        return nullptr;
    }
    void* ptr      = nullptr;
    int   o_device = 0;

    getSetDevice(device_id_, &o_device);
    hipMalloc(&ptr, (size_t)(ceil(size / 32.)) * 32);
    if (is_set_zero) {
        hipMemset(ptr, 0, (size_t)(ceil(size / 32.)) * 32);
    }
    //getSetDevice(o_device);
    std::lock_guard<std::mutex> lock(lock_);
    pointer_mapping_->insert({ptr, size});
    return ptr;
}

void Allocator<AllocatorType::ROCM>::free(void** ptr) {
    void* address = *ptr;
    if (*ptr != nullptr) {
        int o_device = 0;
        std::lock_guard<std::mutex> lock(lock_);
        if (pointer_mapping_->count(address)) {
            getSetDevice(device_id_, &o_device);
            hipFree(*ptr);
            //getSetDevice(o_device);
            pointer_mapping_->erase(address);
        } else {
            FT_LOG_WARNING("pointer_mapping_ does not have information of ptr at %p.", address);
        }
    }
    *ptr = nullptr;
    return;
}

Allocator<AllocatorType::ROCM_HOST>::Allocator(int device_id): PurePointerRocmAllocator(device_id) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
}

Allocator<AllocatorType::ROCM_HOST>::~Allocator() {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    while (!pointer_mapping_->empty()) {
        free((void**)(&pointer_mapping_->begin()->first));
    }
}

void* Allocator<AllocatorType::ROCM_HOST>::malloc(size_t size, const bool is_set_zero) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (size == 0) {
        return nullptr;
    }
    void* ptr      = nullptr;
    int   o_device = 0;

    ptr = std::malloc(size);
    if (is_set_zero) {
        memset(ptr, 0, size);
    }
    FT_LOG_DEBUG("malloc cuda host buffer %p with size %ld", ptr, size);
    std::lock_guard<std::mutex> lock(lock_);
    pointer_mapping_->insert({ptr, size});

    return ptr;
}

void Allocator<AllocatorType::ROCM_HOST>::free(void** ptr) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    void* address = *ptr;
    if (*ptr != nullptr) {
        int o_device = 0;
        std::lock_guard<std::mutex> lock(lock_);
        if (pointer_mapping_->count(address)) {
            FT_LOG_DEBUG("Free buffer %p", address);
            std::free(*ptr);
            pointer_mapping_->erase(address);
        } else {
            FT_LOG_WARNING("pointer_mapping_ does not have information of ptr at %p.", address);
        }
    }
    *ptr = nullptr;
    return;
}

}  // namespace fastertransformer
