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
 
 CudfWrapper::CudfWrapper(cudf::mutable_table_view table_view) {
   mtv = table_view;
 }
 
 void CudfWrapper::tenth_mm_to_inches(int column_index) {
 
  // Example of showing num_columns and num_rows only for potential debugging
  printf("kernel_wrapper.cu # of columns: %lu\n", mtv.num_columns());
  printf("kernel_wrapper.cu # of rows: %lu\n", mtv.num_rows());
 
  std::unique_ptr<cudf::mutable_column_device_view, std::function<void(cudf::mutable_column_device_view*)>> 
        mutable_device_column = cudf::mutable_column_device_view::create(mtv.column(column_index));

  printf("Number rows in mutable_device_column: %lu\n", mutable_device_column->size());

  auto s = cudf::get_element(mtv.column(column_index), 0);
  using ScalarType = cudf::scalar_type_t<int64_t>;
  auto typed_s     = static_cast<ScalarType const *>(s.get());
  printf("Value before kernel: %d\n", typed_s->value());
 
  // Invoke the Kernel to convert tenth_mm -> inches
  kernel_tenth_mm_to_inches<<<1, 1>>>(*mutable_device_column);
  cudaError_t err = cudaStreamSynchronize(0);
  printf("cudaStreamSynchronize Response = %d\n", (int)err);

  s = cudf::get_element(mtv.column(column_index), 0);
  typed_s     = static_cast<ScalarType const *>(s.get());
  printf("Value after kernel: %d\n", typed_s->value()); // Value is same as before
 }
 
 CudfWrapper::~CudfWrapper() {
   // It is important to note that CudfWrapper does not own the underlying Dataframe 
   // object and that will be freed by the Python/Cython layer later.
 }
