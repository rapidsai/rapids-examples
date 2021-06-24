/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "kernel.cuh"

__global__ void kernel_tenth_mm_to_inches(cudf::mutable_column_device_view column)
{
    printf("Please show up!!!\n");
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < column.size()) {
        column.element<int64_t>(i) = 99; // Just for testing purposes
    }
}
