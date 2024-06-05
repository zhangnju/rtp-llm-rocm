# begin /* Add unit tests */
find_package(PythonInterp 3.10 REQUIRED)
find_package(PythonLibs 3.10 REQUIRED)
set(SOURCE  RocmOpsTest.cc 
)

set_source_files_properties(${SOURCE} 
  PROPERTIES 
    LANGUAGE HIP C CXX
)
set(TEST_NAME RocmOpsTest)
add_executable(${TEST_NAME} ${SOURCE} 
    /workspace/rocm-rtp-llm/src/fastertransformer/devices/rocm_impl/RocmDevice.cc 
    /workspace/rocm-rtp-llm/src/fastertransformer/devices/rocm_impl/RocmOps.cc
    /workspace/rocm-rtp-llm/src/fastertransformer/devices/DeviceFactory.cc
    /workspace/rocm-rtp-llm/src/fastertransformer/devices/DeviceBase.cc
    /workspace/rocm-rtp-llm/src/fastertransformer/devices/DeviceOps.cc
    /workspace/rocm-rtp-llm/src/fastertransformer/devices/DeviceBase.cc
    /workspace/rocm-rtp-llm/src/fastertransformer/devices/BufferManager.cc
    /workspace/rocm-rtp-llm/src/fastertransformer/devices/OpData.cc
    /workspace/rocm-rtp-llm/src/fastertransformer/rocm/allocator_rocm.cc
    /workspace/rocm-rtp-llm/src/fastertransformer/core/Buffer.cc
    /workspace/rocm-rtp-llm/src/fastertransformer/core/Types.cc
    /workspace/rocm-rtp-llm/src/fastertransformer/core/allocator.cc
    /workspace/rocm-rtp-llm/src/fastertransformer/utils/ShapeCheck.cc
    /workspace/rocm-rtp-llm/src/fastertransformer/utils/logger.cc
    /workspace/rocm-rtp-llm/src/fastertransformer/utils/exception.cc
    /workspace/rocm-rtp-llm/src/fastertransformer/kernels/hip_kernel/gpt_kernels.hip)
include_directories(/opt/rocm/include/ /usr/include/python3.10/ /workspace/rocm-rtp-llm/)
target_link_libraries(${TEST_NAME} 
    PUBLIC GTest::gtest GTest::gtest_main "${TORCH_LIBRARIES}" pthread pybind11::module
)

set_target_properties(${TEST_NAME}
    PROPERTIES
      CXX_STANDARD 17
      HIP_STANDARD 17
#      HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES}
)

add_test(NAME test_${TEST_NAME} COMMAND ${TEST_NAME})
message(STATUS "Unit Test Added: ${TEST_NAME}")
# end /* Add unit tests */