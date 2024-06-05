#include "src/fastertransformer/devices/rocm_impl/RocmDevice.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/rocm/allocator_rocm.h"
#include "src/fastertransformer/utils/logger.h"

#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include <unistd.h>

namespace fastertransformer {

static const size_t DEFAULT_MAX_BATCH_SIZE = 256;
#define check_cuda_error(val) {} 
RocmDevice::RocmDevice(const DeviceInitParams& params) : DeviceBase(params) {
    FT_LOG_INFO("Initialize RocmDevice. %d", device_id_);
    check_cuda_error(hipSetDevice(device_id_));
    check_cuda_error(hipStreamCreate(&stream_));

    auto allocator_ptr = new Allocator<AllocatorType::ROCM>(device_id_);
    allocator_ptr->setStream(stream_);
    allocator_.reset(allocator_ptr);
    auto host_allocator_ptr = new Allocator<AllocatorType::ROCM_HOST>(device_id_);
    host_allocator_ptr->setStream(stream_);
    host_allocator_.reset(host_allocator_ptr);

    check_cuda_error(hipblasCreate(&hipblas_handle_));
    check_cuda_error(hipblasLtCreate(&hipblaslt_handle_));
    check_cuda_error(hipblasSetStream(hipblas_handle_, stream_));
    check_cuda_error(hipGetDeviceProperties(&device_prop_, device_id_));
/*
    cublas_algo_map_.reset(new cublasAlgoMap(GEMM_CONFIG));
    cublas_mm_wrapper_.reset(new cublasMMWrapper(
        cublas_handle_, cublaslt_handle_, stream_, cublas_algo_map_.get(),
        &cublas_wrapper_mutex_, allocator_.get()));
    cublas_mm_wrapper_->setGemmConfig(HIP_R_16F, HIP_R_16F, HIP_R_16F, HIP_R_32F);

    auto ret = nvmlInit();
    FT_CHECK(ret == NVML_SUCCESS);
    ret = nvmlDeviceGetHandleByIndex(device_id_, &nvml_device_);
    FT_CHECK(ret == NVML_SUCCESS);

    if (params.tp_size > 1) {
        const auto rank = params.tp_rank;
        const auto world_size = params.tp_size;

        nccl_param_.rank_ = rank;
        nccl_param_.world_size_ = world_size;
        auto tcpStore = createTcpStore(
            params.master_ip, params.master_port, world_size, rank);
        const auto nccl_id = &(nccl_param_.nccl_uid_);

        const std::string tp_group_name = "RTP_LLM_TP_GROUP_";
        if (rank == 0) {
            FT_LOG_INFO("rank %d creates nccl uid in group %s.", rank, tp_group_name.c_str());
            NCCLCHECK(ncclGetUniqueId(nccl_id));
            setUniqueId(nccl_id, tp_group_name, tcpStore);
        } else {
            FT_LOG_INFO("rank %d get nccl uid in group %s.", rank, tp_group_name.c_str());
            getUniqueId(nccl_id, tp_group_name, tcpStore);
        }

        FT_LOG_INFO("Initialize NCCL communicators rank %d of %d.", rank, world_size);
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclCommInitRank(&nccl_param_.nccl_comm_, world_size, *nccl_id, rank));
        NCCLCHECK(ncclGroupEnd());
    }
*/ 
}

RocmDevice::~RocmDevice() {
    hiprandstate_buf_.reset();
    //cublas_mm_wrapper_.reset();
    check_cuda_error(hipStreamDestroy(stream_));
    check_cuda_error(hipblasDestroy(hipblas_handle_));
    check_cuda_error(hipblasLtDestroy(hipblaslt_handle_));
    //if (nccl_param_.nccl_comm_) {
    //    ncclCommDestroy(nccl_param_.nccl_comm_);
    //}
}

void RocmDevice::init() {
    DeviceBase::init();
    printf("max batch size: %d\n", init_params_.max_batch_size);
    hiprandstate_buf_ = allocateBuffer({init_params_.max_batch_size * sizeof(hiprandState_t)});
}

void RocmDevice::syncAndCheck() {
    //syncCommunication();
    hipDeviceSynchronize();
    //sync_check_cuda_error();
}
/*
void CudaDevice::syncCommunication() {
    if (nccl_param_.world_size_ > 1) {
        FT_LOG_INFO("Synchronize NCCL communicators rank %d of %d.", nccl_param_.rank_, nccl_param_.world_size_);
        ftNcclStreamSynchronize(nccl_param_, stream_);
    }
}
*/
DeviceProperties RocmDevice::getDeviceProperties() {
    static DeviceProperties* prop = nullptr;
    if (prop == nullptr) {
        prop = new DeviceProperties();
        prop->type = DeviceType::Rocm;
        prop->id = device_id_;
        //prop->tp_rank = nccl_param_.rank_;
        //prop->tp_size = nccl_param_.world_size_;
    }
    return *prop;
}

// TODO(wangyin.yx): fill all memory status.
DeviceStatus RocmDevice::getDeviceStatus() {
    DeviceStatus status;

    size_t total_bytes;
    auto error = hipMemGetInfo(&status.device_memory_status.free_bytes, &total_bytes);
    FT_CHECK(error == hipSuccess);
    status.device_memory_status.used_bytes = total_bytes - status.device_memory_status.free_bytes;

    const auto buffer_status = queryBufferStatus();
    status.device_memory_status.allocated_bytes = buffer_status.device_allocated_bytes;
    status.device_memory_status.preserved_bytes = buffer_status.device_preserved_bytes;
    status.host_memory_status.allocated_bytes = buffer_status.host_allocated_bytes;

    //nvmlUtilization_t utilization;
    //auto ret = nvmlDeviceGetUtilizationRates(nvml_device_, &utilization);
    //FT_CHECK(ret == NVML_SUCCESS);
    //status.device_utilization = (float)utilization.gpu;

    return status;
}

RTP_LLM_REGISTER_DEVICE(Rocm);

}; // namespace fastertransformer

