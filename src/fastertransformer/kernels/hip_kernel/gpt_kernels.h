/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <unordered_map>

//#include "src/fastertransformer/core/Tensor.h"
//#include "src/fastertransformer/cuda/memory_utils.h"

namespace fastertransformer {

template<typename T>
void invokeTransposeAxis012(T* out, T* in, const int dim0, const int dim1, const int dim2, hipStream_t stream);

// from [b, s, h, d] to [b, h, s, d]
template<typename T>
void invokeTransposeAxis12(T* out, T* in, const int dim0, const int dim1, const int dim2, const int dim_3, hipStream_t stream);

template<typename T>
void invokeTransposeAxis01(
    T* out, T* in, const int dim0, const int dim1, hipStream_t stream);

}  // namespace fastertransformer
