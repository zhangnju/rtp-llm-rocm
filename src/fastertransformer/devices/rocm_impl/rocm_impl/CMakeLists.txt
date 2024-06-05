################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc.
################################################################################

cmake_minimum_required (VERSION 3.21)

set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "")

project(rtp-llm-rocm-test LANGUAGES CXX C HIP)

add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")
set(CMAKE_BUILD_TYPE "Release")
set(PYBIND11_CPP_STANDARD -std=c++11)
 
 
find_package(pybind11 REQUIRED)
 
INCLUDE_DIRECTORIES(
  ${pybind11_INCLUDE_DIRS}
)

enable_testing()
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR})
find_package(GTest REQUIRED)
find_package(Torch REQUIRED)
add_subdirectory(RocmOpsTest)

