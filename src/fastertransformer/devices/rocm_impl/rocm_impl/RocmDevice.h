#pragma once

#include "src/fastertransformer/devices/DeviceBase.h"
//#include "src/fastertransformer/cuda/cuda_utils.h"
//#include "src/fastertransformer/cuda/cublas/cublas.h"
//#include "src/fastertransformer/cuda/nccl/nccl_utils.h"

//#include <nvml.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

namespace fastertransformer {

class RocmDevice : public DeviceBase {
public:
    RocmDevice(const DeviceInitParams& params);
    ~RocmDevice();

public:
    void init() override;
    DeviceProperties getDeviceProperties() override;
    DeviceStatus getDeviceStatus() override;
    IAllocator* getAllocator() override { return allocator_.get(); }
    IAllocator* getHostAllocator() override { return host_allocator_.get(); }

    void syncAndCheck() override;
    /*
    void syncCommunication() override;
    */
public:
    hipStream_t getStream() {return stream_;}
    //NcclParam getNcclParam() {return nccl_param_;}
public:
    void copy(const CopyParams& params);
    TransposeOutput transpose(const TransposeParams& params);
    //ConvertOutput convert(const ConvertParams& params);
    /*
    LayernormOutput layernorm(const LayernormParams& params);
    BufferPtr gemm(const GemmParams& params);
    BufferPtr embeddingLookup(const EmbeddingLookupParams& params);
    void activation(const ActivationParams& params);
    BufferPtr softmax(const SoftmaxParams& params);
    AttentionModuleOutput contextAttention(const AttentionModuleParams& params);
    AttentionModuleOutput decoderSelfAttention(const AttentionModuleParams& params);
    void sampleGreedy(const GreedyParams& params);
    */
    //void broadcast(const BroadcastParams& params);
    //void allReduce(const AllReduceParams& params);
    //void allGather(const AllGatherParams& params);

// TODO: @xinglai delelte this
public:
    hipStream_t stream() const {return stream_;}
//    cublasMMWrapper* cublasMMWrapperPtr() const {return cublas_mm_wrapper_.get();}

private:
    std::unique_ptr<IAllocator> allocator_;
    std::unique_ptr<IAllocator> host_allocator_;

    hipStream_t stream_;
    hipblasHandle_t hipblas_handle_;
    hipblasLtHandle_t hipblaslt_handle_;
    hipDeviceProp_t device_prop_;

    std::mutex hipblas_wrapper_mutex_;
    //std::unique_ptr<cublasAlgoMap> cublas_algo_map_;
    //std::unique_ptr<cublasMMWrapper> cublas_mm_wrapper_;

    //nvmlDevice_t nvml_device_;
    //NcclParam nccl_param_;

    BufferPtr hiprandstate_buf_; // for sampler use.
};

} // namespace fastertransformer

