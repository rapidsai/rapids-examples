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
 #include <sstream>
 #include <string>
 #include <vector>
 
 #include <rmm/thrust_rmm_allocator.h>
 #include <cudf/column/column_device_view.cuh>
 #include <cudf/column/column_factories.hpp>
 #include <cudf/io/csv.hpp>
 #include <cudf/strings/string_view.cuh>
 #include <cudf/strings/strings_column_view.hpp>
 #include <cudf/table/table.hpp>
 #include <cudf/table/table_view.hpp>
 
 #include <locale.h>
 #include <thrust/device_vector.h>
 #include <thrust/execution_policy.h>
 #include <thrust/for_each.h>
 #include <thrust/transform.h>
 #include <unistd.h>
 
 #include "dstring.cuh"
 #include "jitify.hpp"
 
 double GetTime()
 {
   timeval tv;
   gettimeofday(&tv, NULL);
   return (double)(tv.tv_sec * 1000000 + tv.tv_usec) / 1000000.0;
 }
 
 using string_index_pair = thrust::pair<const char*, cudf::size_type>;
 
 rmm::device_vector<dstring_view> create_vector_from_column(cudf::strings_column_view const& strings)
 {
   auto strings_column = cudf::column_device_view::create(strings.parent());
   auto d_column       = *strings_column;
   auto count          = strings.size();
   rmm::device_vector<dstring_view> strings_vector(count);
   thrust::transform(thrust::device,
                     thrust::make_counting_iterator<cudf::size_type>(0),
                     thrust::make_counting_iterator<cudf::size_type>(count),
                     strings_vector.begin(),
                     [d_column] __device__(cudf::size_type idx) {
                       if (d_column.is_null(idx)) return dstring_view(nullptr, 0);
                       auto d_str = d_column.template element<cudf::string_view>(idx);
                       if (d_str.empty()) return dstring_view{"", 0};
                       return dstring_view{d_str.data(), d_str.size_bytes()};
                     });
   return strings_vector;
 }
 
 std::string load_udf(std::ifstream& input)
 {
   std::stringstream udf;
   udf << "\n#include \"dstring.cuh\"\n";
   std::string line;
   while (std::getline(input, line)) udf << line << "\n";
   return udf.str();
 }
 
 void write_output(std::ostream& stream, const char* output, const int* offsets, unsigned int count)
 {
   for (unsigned int idx = 0; idx < count; ++idx) {
     int offset      = offsets[idx];
     const char* str = output + offset;
     int length      = offsets[idx + 1] - offset;
     stream.write(str, length);
     stream << "\n";
   }
 }
 
 std::map<std::string, std::string> parse_cli_parms(int argc, const char** argv)
 {
   std::map<std::string, std::string> parms;
   while (argc > 1) {
     const char* value = argv[argc - 1];
     const char* key   = (argv[argc - 2]) + 1;
     parms[key]        = value;
     argc -= 2;
   }
   return parms;
 }
 
 int main(int argc, const char** argv)
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
   int column    = 0;
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

  cudf::io::csv_reader_options in_args = cudf::io::csv_reader_options::builder(cudf::io::source_info{csv_file})
                                          .header(-1)
                                          .use_cols_indexes(std::vector<int>{ column })
                                          .dtypes(std::vector<std::string>{ "str" })
                                          .nrows( rows > 0 ? rows : 0)
                                          .build();

  auto csv_result = cudf::io::read_csv(in_args);

   auto strs       = cudf::strings_column_view(csv_result.tbl->view().column(0));
 
   auto strings_count = strs.size();
   printf("strings count = %d\n", strings_count);
   std::string udf = load_udf(udf_stream);
 
   // setup malloc heap size
   size_t heap_size      = 1024;  // 1GB;
   std::string heap_parm = parms["m"];
   if (!heap_parm.empty()) heap_size = std::atoi(heap_parm.c_str());
   heap_size *= 1024 * 1024;
   size_t max_malloc_heap_size = 0;
   cudaDeviceGetLimit(&max_malloc_heap_size, cudaLimitMallocHeapSize);
   if (max_malloc_heap_size < heap_size) max_malloc_heap_size = heap_size;
   if (cudaDeviceSetLimit(cudaLimitMallocHeapSize, max_malloc_heap_size) != cudaSuccess) {
     fprintf(stderr, "could not set malloc heap size to %ldMB\n", (heap_size / (1024 * 1024)));
     return -1;
   }
 
   // create input array of dstring_view objects
   rmm::device_vector<dstring_view> in_strs = create_vector_from_column(strs);
   auto d_in_strs                           = in_strs.data().get();
 
   // allocate an output array
   thrust::device_vector<dstring> out_strs(strings_count);
   dstring* d_out_strs = out_strs.data().get();
 
   double et_load_data = GetTime() - st_load_data;
   if (parms.find("v") != parms.end()) fprintf(stderr, "Load data: %g seconds\n", et_load_data);
 
   // launch custom kernel
   {
     double st_kernel = GetTime();
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
     const char* result_str = "ok";
     if (result) cuGetErrorName(result, &result_str);
     fprintf(stderr, "launch result = %d [%s]\n", (int)result, result_str);
     fprintf(stderr, "%s=(%d) ", udf_name.c_str(), (int)cudaDeviceSynchronize());
     double et_kernel = GetTime() - st_kernel;
     fprintf(stderr, "%g seconds\n", et_kernel);
   }
 
   double st_output_data = GetTime();
 
   // convert output to pointers array
   rmm::device_vector<string_index_pair> ptrs(strings_count);
   thrust::transform(thrust::device,
                     out_strs.begin(),
                     out_strs.end(),
                     ptrs.begin(),
                     [] __device__(auto const& dstr) {
                       return string_index_pair{dstr.data(), (cudf::size_type)dstr.size_bytes()};
                     });
   auto results = cudf::make_strings_column(ptrs);
 
   double et_output_data = GetTime() - st_output_data;
   if (parms.find("v") != parms.end())
     fprintf(stderr, "Create strings column: %g seconds\n", et_output_data);
 
   // create column from output array
   auto scv       = cudf::strings_column_view(results->view());
   st_output_data = GetTime();
 
   // output results
   std::string out_filename = parms["f"];
   if (out_filename.empty()) {
     cudf::strings::print(scv);
     return 0;
   }
 
   // write csv file
   auto output_table = cudf::table_view{std::vector<cudf::column_view>{results->view()}};
   cudf::io::sink_info const sink{out_filename};
  cudf::io::csv_writer_options write_args = cudf::io::csv_writer_options::builder(sink, output_table)
                                            .na_rep(std::string(""))
                                            .include_header(false)
                                            .rows_per_chunk(1)
                                            .build();
  cudf::io::write_csv(write_args);
 
   et_output_data = GetTime() - st_output_data;
   if (parms.find("v") != parms.end())
     fprintf(stderr, "Output to file: %g seconds\n", et_output_data);
 
   return 0;
 }