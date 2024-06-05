#pragma once
#include "src/fastertransformer/core/allocator.h"
//#include "src/fastertransformer/cuda/cuda_utils.h"
#include <mutex>
#include <hip/hip_runtime.h>

namespace fastertransformer{
#define FT_CHECK(val) {}
class IRocmAllocator: virtual public IAllocator {
public:
    IRocmAllocator(int device_id)
    : device_id_(device_id) {}
    virtual ~IRocmAllocator() {};

    MemoryType memoryType() const override {
        return MEMORY_GPU;
    }

    void setStream(hipStream_t stream) {
        stream_ = stream;
    }

    hipStream_t returnStream() {
        return stream_;
    };

    void* reMalloc(void* ptr, size_t size, const bool is_set_zero = false) override;

    void memSet(void* ptr, const int val, const size_t size) const override;

protected:
    virtual bool isExist(void* address) const = 0;
    virtual ReallocType isReMalloc(void* address, size_t size) const = 0;

protected:
    hipStream_t                                       stream_ = 0;  // initialize as default stream
    const int                                          device_id_;
};

class PurePointerRocmAllocator: public IRocmAllocator {
public:
    PurePointerRocmAllocator(int device_id)
    : IRocmAllocator(device_id)
    , pointer_mapping_(new std::unordered_map<void*, size_t>)
    {}
    ~PurePointerRocmAllocator() {}

protected:
    virtual bool isExist(void* address) const {
        return pointer_mapping_->count(address) > 0;
    }

    virtual ReallocType isReMalloc(void* address, size_t size) const {
        FT_CHECK(isExist(address));
        if (pointer_mapping_->at(address) < size) {
            return ReallocType::INCREASE;
        } else if (pointer_mapping_->at(address) == size) {
            return ReallocType::REUSE;
        } else {
            return ReallocType::DECREASE;
        }
    }

protected:
    std::unique_ptr<std::unordered_map<void*, size_t>> pointer_mapping_;
    std::mutex lock_;
};

template<>
class Allocator<AllocatorType::ROCM>: public PurePointerRocmAllocator, public TypedAllocator<AllocatorType::ROCM> {
public:
    Allocator(int device_id);
    ~Allocator();

    void* malloc(size_t size, const bool is_set_zero = false) override;
    void free(void** ptr) override;
};

template<>
class Allocator<AllocatorType::ROCM_HOST> : public PurePointerRocmAllocator, public TypedAllocator<AllocatorType::ROCM_HOST>{
public:
    Allocator(int device_id);
    ~Allocator();

    MemoryType memoryType() const override {
        return MEMORY_CPU_PINNED;
    }

    void* malloc(size_t size, const bool is_set_zero = false) override;
    void free(void** ptr) override;
};


} // namespace fastertransformer

