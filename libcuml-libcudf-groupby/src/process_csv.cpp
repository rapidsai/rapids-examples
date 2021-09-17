#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/copying.hpp>
#include <cudf/reshape.hpp>
#include <cudf/transpose.hpp>
#include <cudf/table/table.hpp>

#include <cuml/linear_model/glm.hpp>
#include <raft/handle.hpp>
#include <raft/cudart_utils.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include <rmm/exec_policy.hpp>
#include <cuda_runtime.h>

#include <thrust/uninitialized_fill.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>




#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                    \
  {                                                                           \
    cudaError_t cudaStatus = call;                                            \
    if (cudaSuccess != cudaStatus)                                            \
      fprintf(stderr,                                                         \
              "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with " \
              "%s (%d).\n",                                                   \
              #call,                                                          \
              __LINE__,                                                       \
              __FILE__,                                                       \
              cudaGetErrorString(cudaStatus),                                 \
              cudaStatus);                                                    \
  }
#endif  // CUDA_RT_CALL


cudf::io::table_with_metadata read_csv(std::string const& file_path)
{
  auto source_info = cudf::io::source_info(file_path);
  auto builder     = cudf::io::csv_reader_options::builder(source_info);
  auto options     = builder.build();
  return cudf::io::read_csv(options);
}

void write_csv(cudf::table_view const& tbl_view, std::string const& file_path)
{
  auto sink_info = cudf::io::sink_info(file_path);
  auto builder   = cudf::io::csv_writer_options::builder(sink_info, tbl_view);
  auto options   = builder.build();
  cudf::io::write_csv(options);
}


std::unique_ptr<cudf::column> generate_grouped_arr(cudf::table_view values, cudf::size_type start, cudf::size_type end)
{
  auto sliced_table = cudf::slice(values, {start, end}).front();
  auto [_, transposed_table] = cudf::transpose(sliced_table);

  return cudf::interleave_columns(transposed_table);
}

std::unique_ptr<cudf::table> cuml_regression_on_groupby(cudf::table_view input_table)
{
  // Schema: | Timestamp | Name | X | Y
  auto keys = cudf::table_view{{input_table.column(0)}};  // name

  cudf::groupby::groupby grpby_obj(keys);
  cudf::groupby::groupby::groups gb_groups = grpby_obj.get_groups(input_table.select({1,2}));
  auto values_view = (gb_groups.values)->view();
  
  auto interleaved = generate_grouped_arr(values_view, 0, 3);

  // cuml setup
  int n_cols = 2;
  raft::handle_t handle;
  cudaStream_t stream = rmm::cuda_stream_default.value();
  CUDA_RT_CALL(cudaStreamCreate(&stream));
  handle.set_stream(stream);

  // looping through each group
  for (int i = 1; i < gb_groups.offsets.size(); i++) {
    cudf::size_type offset1 = gb_groups.offsets[i-1], offset2 = gb_groups.offsets[i];
    auto interleaved = generate_grouped_arr(values_view, offset1, offset2);
    double *matrix_pointer = interleaved->mutable_view().data<double>();

    // original values
    raft::print_device_vector<double>("values", matrix_pointer, (offset2 - offset1) * n_cols, std::cout);

    int n_rows = (offset2 - offset1) * n_cols;
    thrust::device_ptr<double> labels = thrust::device_malloc<double>(n_rows);
    thrust::device_ptr<double> coef = thrust::device_malloc<double>(n_cols);
    double intercept;
    ML::GLM::olsFit(handle, matrix_pointer, n_rows, n_cols, labels.get(), coef.get(), &intercept, false, false);

    // values overrwritten by olsFit (if the line above is commented out then the same value will be written out)
    raft::print_device_vector<double>("values", matrix_pointer, (offset2 - offset1) * n_cols, std::cout);
  }

  return std::make_unique<cudf::table>(cudf::table_view({interleaved->view()}).select({0}));
}

int main(int argc, char** argv)
{
  // Read data
  auto sample_table = read_csv("test2.csv");

  // Process
  auto result = cuml_regression_on_groupby(*sample_table.tbl);

  // Write out result
  write_csv(*result, "test_out.csv");

  return 0;
}
