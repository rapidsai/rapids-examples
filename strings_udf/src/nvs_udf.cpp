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

#include <cuda_runtime.h>
#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <locale.h>
#include <unistd.h>
#include <memory>

#include "dstring_view.hpp"
#include "jitify.hpp"

double GetTime()
{
  timeval tv;
  gettimeofday(&tv, NULL);
  return (double)(tv.tv_sec * 1000000 + tv.tv_usec) / 1000000.0;
}

std::string load_udf(std::ifstream &input)
{
  std::stringstream udf;
  std::string line;
  while (std::getline(input, line)) udf << line << "\n";
  return udf.str();
}

void print_column(cudf::strings_column_view const &input)
{
  if (input.chars_size() == 0) {
    printf("empty\n");
    return;
  }

  auto offsets = input.offsets();
  std::vector<int32_t> h_offsets(offsets.size());
  auto chars = input.chars();
  std::vector<char> h_chars(chars.size());
  cudaMemcpy(h_offsets.data(),
             offsets.data<int32_t>(),
             offsets.size() * sizeof(int32_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_chars.data(), chars.data<char>(), chars.size(), cudaMemcpyDeviceToHost);

  for (int idx = 0; idx < input.size(); ++idx) {
    int offset      = h_offsets[idx];
    const char *str = h_chars.data() + offset;
    int length      = h_offsets[idx + 1] - offset;
    std::string output(str, length);
    std::cout << output << "\n";
  }
}

std::map<std::string, std::string> parse_cli_parms(int argc, const char **argv)
{
  std::map<std::string, std::string> parms;
  while (argc > 1) {
    const char *value = argv[argc - 1];
    const char *key   = (argv[argc - 2]) + 1;
    parms[key]        = value;
    argc -= 2;
  }
  return parms;
}

std::unique_ptr<cudf::column> process_udf(std::string udf,
                                          std::string udf_name,
                                          cudf::size_type size,
                                          std::vector<cudf::column_view> input,
                                          size_t heap_size)
{
  size_t max_malloc_heap_size = 0;
  cudaDeviceGetLimit(&max_malloc_heap_size, cudaLimitMallocHeapSize);
  if (max_malloc_heap_size < heap_size) {
    max_malloc_heap_size = heap_size;
    if (cudaDeviceSetLimit(cudaLimitMallocHeapSize, max_malloc_heap_size) != cudaSuccess) {
      fprintf(stderr, "could not set malloc heap size to %ldMB\n", (heap_size / (1024 * 1024)));
      return nullptr;
    }
  }

  rmm::cuda_stream_view stream = rmm::cuda_stream_default;

  // create input array of dstring_view objects
  auto strs               = cudf::strings_column_view(input[0]);
  auto strings_count      = strs.size();
  auto in_strs            = cudf::strings::detail::create_string_vector_from_column(strs);
  dstring_view *d_in_strs = reinterpret_cast<dstring_view *>(in_strs.data());

  // allocate an output array
  rmm::device_uvector<cudf::string_view> out_strs(strings_count, stream);
  cudaMemset(out_strs.data(), 0, sizeof(cudf::string_view) * strings_count);

  // add dstring header to udf
  udf = "\n#include \"dstring.cuh\"\n" + udf;

  // launch custom kernel
  {
    auto d_out_strs = reinterpret_cast<dstring *>(out_strs.data());
    // double st_kernel = GetTime();
    static jitify::JitCache kernel_cache;
    // nvrtc did not recognize --expt-relaxed-constexpr
    // also it has trouble including thrust headers
    jitify::Program program = kernel_cache.program(udf.c_str(), 0, {"-I.", "-std=c++14"});
    unsigned num_blocks     = ((strings_count - 1) / 128) + 1;
    dim3 grid(num_blocks);
    dim3 block(128);
    CUresult result = program.kernel(udf_name.c_str())
                        .instantiate()
                        .configure(grid, block)
                        .launch(d_in_strs, d_out_strs, strings_count);
    const char *result_str = "ok";
    if (result) {
      cuGetErrorName(result, &result_str);
      fprintf(stderr, "launch result = %d [%s]\n", (int)result, result_str);
    }
    auto err = cudaDeviceSynchronize();
    if (err) { fprintf(stderr, "%s=(%d) ", udf_name.c_str(), (int)err); }
    // double et_kernel = GetTime() - st_kernel;
    // fprintf(stderr, "%g seconds\n", et_kernel);
  }

  // convert the output array into a strings column
  cudf::device_span<cudf::string_view> indices(out_strs.data(), strings_count);
  auto results = cudf::make_strings_column(indices, cudf::string_view(nullptr, 0));
  return results;
}

int main(int argc, const char **argv)
{
  if (argc < 3) {
    printf("parameters:\n");
    printf("-u udf-text-file\n");
    printf("-n kernel-name (default is 'udf')\n");
    printf("-t text/csv-file\n");
    printf("-c 0-based column number if csv file (default is 0=first column)\n");
    printf("-r number of rows to read from file (default is 0=entire file)\n");
    printf("-f output file (default is stdout)\n");
    printf("-m malloc heap size (default is 1GB)\n");
    return 0;
  }

  std::map<std::string, std::string> parms = parse_cli_parms(argc, argv);

  std::string udf_text = parms["u"];
  std::string csv_file = parms["t"];
  if (udf_text.empty() || csv_file.empty()) {
    printf("udf file (-u) and text file (-t) are required parameters.\n");
    return 0;
  }
  std::string udf_name = parms["n"];
  if (udf_name.empty()) udf_name = "udf";
  std::string csv_column = parms["c"];
  unsigned int column    = 0;
  if (!csv_column.empty()) column = std::atoi(csv_column.c_str());
  std::string csv_rows = parms["r"];
  unsigned int rows    = 0;
  if (!csv_rows.empty()) rows = std::atoi(csv_rows.c_str());
  //
  std::ifstream udf_stream(udf_text);
  if (!udf_stream.is_open()) {
    printf("could not open file [%s]\n", udf_text.c_str());
    return 0;
  }

  double st_load_data = GetTime();
  cudf::io::csv_reader_options in_opts =
    cudf::io::csv_reader_options::builder(cudf::io::source_info{csv_file}).header(-1);
  in_opts.set_use_cols_indexes({(int)column});
  if (rows > 0) in_opts.set_nrows((int)rows);
  in_opts.set_dtypes({"str"});
  auto csv_result = cudf::io::read_csv(in_opts);
  auto strs       = cudf::strings_column_view(csv_result.tbl->view().column(0));

  double et_load_data = GetTime() - st_load_data;
  if (parms.find("v") != parms.end()) fprintf(stderr, "Load data: %g seconds\n", et_load_data);

  auto strings_count = strs.size();
  printf("strings count = %d\n", strings_count);
  std::string udf = load_udf(udf_stream);

  // setup malloc heap size
  size_t heap_size      = 1024;  // 1GB;
  std::string heap_parm = parms["m"];
  if (!heap_parm.empty()) heap_size = std::atoi(heap_parm.c_str());
  heap_size *= 1024 * 1024;

  double st_output_data = GetTime();

  auto results = process_udf(udf, udf_name, strings_count, {strs.parent()}, heap_size);

  double et_output_data = GetTime() - st_output_data;
  if (parms.find("v") != parms.end())
    fprintf(stderr, "Create strings column: %g seconds\n", et_output_data);

  // create column from output array
  auto scv       = cudf::strings_column_view(results->view());
  st_output_data = GetTime();

  // output results
  std::string out_filename = parms["f"];
  if (out_filename.empty()) {
    print_column(scv);
    return 0;
  }

  // write csv file
  auto output_table = cudf::table_view{std::vector<cudf::column_view>{results->view()}};
  cudf::io::sink_info const sink{out_filename};
  cudf::io::csv_writer_options writer_options =
    cudf::io::csv_writer_options::builder(sink, output_table).include_header(false);
  cudf::io::write_csv(writer_options);

  et_output_data = GetTime() - st_output_data;
  if (parms.find("v") != parms.end())
    fprintf(stderr, "Output to file: %g seconds\n", et_output_data);

  return 0;
}
