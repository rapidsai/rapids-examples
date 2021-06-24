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

#include "cudf/utilities/type_dispatcher.hpp"
#include "kernel_wrapper.hpp"

static constexpr double mm_to_inches = 0.0393701;

__global__ void kernel_tenth_mm_to_inches(cudf::mutable_column_device_view column)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < column.size()) {
      column.element<double>(i) = column.element<double>(i) * mm_to_inches;
    }
}
 
 CudfWrapper::CudfWrapper(cudf::mutable_table_view table_view) {
   mtv = table_view;
 }
 
 void CudfWrapper::tenth_mm_to_inches(int column_index) {
 
  // Example of showing num_columns and num_rows only for potential debugging
  printf("kernel_wrapper.cu # of columns: %lu\n", mtv.num_columns());
  printf("kernel_wrapper.cu # of rows: %lu\n", mtv.num_rows());
 
  std::unique_ptr<cudf::mutable_column_device_view, std::function<void(cudf::mutable_column_device_view*)>> 
        mutable_device_column = cudf::mutable_column_device_view::create(mtv.column(column_index));

  // If you need to get the value of an individual element from host code use the below code snippet
  auto s = cudf::get_element(mtv.column(column_index), 5);
  using ScalarType = cudf::scalar_type_t<double>;
  auto typed_s     = static_cast<ScalarType const *>(s.get());
  printf("Value before kernel: %f\n", typed_s->value());
 
  // Invoke the Kernel to convert tenth_mm -> inches
  kernel_tenth_mm_to_inches<<<(mtv.num_rows()+255)/256, 256>>>(*mutable_device_column);
  cudaError_t err = cudaStreamSynchronize(0);
  printf("cudaStreamSynchronize Response = %d\n", (int)err);
 }
 
 CudfWrapper::~CudfWrapper() {
   // It is important to note that CudfWrapper does not own the underlying Dataframe 
   // object and that will be freed by the Python/Cython layer later.
 }
