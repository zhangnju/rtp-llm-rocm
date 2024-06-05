/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <assert.h>
#include <hip/hip_runtime.h>

#include "src/fastertransformer/kernels/hip_kernel/gpt_kernels.h"
//#include "src/fastertransformer/cuda/memory_utils.h"

namespace fastertransformer {

// TODO Add half2 implementation
template<typename T>
__global__ void transposeAxis01(T* out, T* in, const int dim0, const int dim1, const int dim2)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < dim0 * dim1 * dim2) {
        const int input_dim2_index = index % dim2;
        index                      = (index - input_dim2_index) / dim2;
        const int input_dim1_index = index % dim1;
        index                      = (index - input_dim1_index) / dim1;
        const int input_dim0_index = index % dim0;

        out[input_dim1_index * dim0 * dim2 + input_dim0_index * dim2 + input_dim2_index] =
            in[input_dim0_index * dim1 * dim2 + input_dim1_index * dim2 + input_dim2_index];
    }
}

template<typename T>
void invokeTransposeAxis012(T* out, T* in, const int dim0, const int dim1, const int dim2, hipStream_t stream)
{
    dim3 block(512);
    dim3 grid((int)(ceil(dim0 * dim1 * dim2 / 512.)));
    transposeAxis01<<<grid, block, 0, stream>>>(out, in, dim0, dim1, dim2);
}

template void
invokeTransposeAxis012(float* out, float* in, const int dim0, const int dim1, const int dim2, hipStream_t stream);

template void
invokeTransposeAxis012(half* out, half* in, const int dim0, const int dim1, const int dim2, hipStream_t stream);

#ifdef ENABLE_BF16
template void
invokeTransposeAxis012(hip_bfloat16* out, hip_bfloat16* in, const int dim0, const int dim1, const int dim2, hipStream_t stream);
#endif

template void
invokeTransposeAxis012(int* out, int* in, const int dim0, const int dim1, const int dim2, hipStream_t stream);

template<typename T>
__global__ void transposeAxis01(T* out, T* in, const int* in_skipping_dim1, const int dim0, const int dim1)
{
    // out: [dim1, dim0]
    // in: [dim0, dim1]
    // in_skipping_dim1: [dim1]

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < dim0 * dim1) {
        const int input_dim1_index = index % dim1;
        index                      = (index - input_dim1_index) / dim1;
        const int input_dim0_index = index % dim0;
        const int in_offset        = in_skipping_dim1 == nullptr ? 0 : in_skipping_dim1[input_dim1_index] * dim1;

        out[input_dim1_index * dim0 + input_dim0_index] = in[in_offset + input_dim0_index * dim1 + input_dim1_index];
    }
}

template<typename T>
void invokeTransposeAxis01(
    T* out, T* in, const int dim0, const int dim1, hipStream_t stream)
{
    dim3 block(512);
    dim3 grid((int)(ceil(dim0 * dim1 / 512.)));
    transposeAxis01<<<grid, block, 0, stream>>>(out, in, nullptr, dim0, dim1);
}

template void invokeTransposeAxis01(
    int32_t* out, int32_t* in, const int dim0, const int dim1, hipStream_t stream);

template void invokeTransposeAxis01(
    int8_t* out, int8_t* in, const int dim0, const int dim1, hipStream_t stream);

template void invokeTransposeAxis01(
    uint8_t* out, uint8_t* in, const int dim0, const int dim1, hipStream_t stream);

template void invokeTransposeAxis01(
    uint32_t* out, uint32_t* in, const int dim0, const int dim1, hipStream_t stream);

template void invokeTransposeAxis01(
    int64_t* out, int64_t* in, const int dim0, const int dim1, hipStream_t stream);

template void invokeTransposeAxis01(
    uint64_t* out, uint64_t* in, const int dim0, const int dim1, hipStream_t stream);

template void invokeTransposeAxis01(
    float* out, float* in, const int dim0, const int dim1, hipStream_t stream);

template void invokeTransposeAxis01(
    half* out, half* in, const int dim0, const int dim1, hipStream_t stream);

#ifdef ENABLE_BF16
template void invokeTransposeAxis01(
    hip_bfloat16* out, hip_bfloat16* in, const int dim0, const int dim1, hipStream_t stream);
#endif

template<typename T>
__global__ void transposeAxis12(T* out, T* in, const int dim0, const int dim1, const int dim2, const int dim3)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < dim0 * dim1 * dim2 * dim3) {
        const int input_dim3_index = index % dim3;
        index                      = (index - input_dim3_index) / dim3;
        const int input_dim2_index = index % dim2;
        index                      = (index - input_dim2_index) / dim2;
        const int input_dim1_index = index % dim1;
        index                      = (index - input_dim1_index) / dim1;
        const int input_dim0_index = index % dim0;
        out[input_dim0_index * dim1 * dim2 * dim3 + input_dim2_index * dim1 * dim3 + input_dim1_index * dim3 + input_dim3_index] =
            in[input_dim0_index * dim1 * dim2 * dim3 + input_dim1_index * dim2 * dim3 + input_dim2_index * dim3 + input_dim3_index];
    }
}

template<typename T>
void invokeTransposeAxis12(T* out, T* in, const int dim0, const int dim1, const int dim2, const int dim_3, hipStream_t stream)
{
    dim3 block(512);
    dim3 grid((int)(ceil(dim0 * dim1 * dim2 * dim_3 / 512.)));
    transposeAxis12<<<grid, block, 0, stream>>>(out, in, dim0, dim1, dim2, dim_3);
}

template void
invokeTransposeAxis12(float* out, float* in, const int dim0, const int dim1, const int dim2, const int dim_3, hipStream_t stream);

template void
invokeTransposeAxis12(half* out, half* in, const int dim0, const int dim1, const int dim2, const int dim_3, hipStream_t stream);

template void
invokeTransposeAxis12(int* out, int* in, const int dim0, const int dim1, const int dim2, const int dim_3, hipStream_t stream);

#ifdef ENABLE_BF16
template void
invokeTransposeAxis12(hip_bfloat16* out, hip_bfloat16* in, const int dim0, const int dim1, const int dim2, const int dim_3, hipStream_t stream);
#endif

}  // namespace fastertransformer
