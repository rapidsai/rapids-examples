#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/copying.hpp>
#include <cudf/reshape.hpp>
#include <cudf/transpose.hpp>
#include <cudf/table/table.hpp>

#include <cuml/linear_model/glm.hpp>
#include <raft/handle.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

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

  // Compute the average of each company's closing price with entire column
  cudf::groupby::groupby grpby_obj(keys);

  cudf::groupby::groupby::groups gb_groups = grpby_obj.get_groups(input_table.select({1,2}));

  auto values_view = (gb_groups.values)->view();
  auto interleaved = generate_grouped_arr(values_view, 0, 3);

  raft::handle_t handle;
  cudaStream_t stream = rmm::cuda_stream_default.value();
  CUDA_RT_CALL(cudaStreamCreate(&stream));
  handle.set_stream(stream);

  // matrix pointer
  float *matrix_pointer = interleaved->mutable_view().data<float>();
  int n_rows = 3;
  int n_cols = 2;

/*
  rmm::device_uvector<float> labels(3, rmm::cuda_stream_default);
  thrust::uninitialized_fill(thrust::cuda::par.on(stream), labels.begin(), labels.end(), 0);

  rmm::device_uvector<float> coef(2, rmm::cuda_stream_default);
  thrust::uninitialized_fill(thrust::cuda::par.on(stream), coef.begin(), coef.end(), 0); 

  rmm::device_uvector<float> intercept(1, rmm::cuda_stream_default);
  thrust::uninitialized_fill(thrust::cuda::par.on(stream), intercept.begin(), intercept.end(), 0);   
*/
  thrust::device_ptr<float> labels = thrust::device_malloc<float>(10);
  thrust::device_ptr<float> coef = thrust::device_malloc<float>(10);
  // thrust::device_ptr<float> intercept = thrust::device_malloc<float>(10);
  thrust::size_type intercept_size = 1;
  thrust::device_vector<float> intercept(1, 0.0);
  float *intercept_ptr = thrust::raw_pointer_cast(intercept.data());


  bool fit_intercept = false;
  bool normalize = false;

  ML::GLM::olsFit(handle, matrix_pointer, n_rows, n_cols, labels.get(), coef.get(), intercept_ptr, fit_intercept, normalize);
  
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
