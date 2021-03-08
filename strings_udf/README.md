# strings_udf
User Defined Function (UDF) prototype for operating against a libcudf strings column.

UDFs are custom kernel/device CUDA code that execute in parallel threads on an NVIDIA GPU.
The libcudf library contains fixed operation primitives written in CUDA/C++ for strings.
Strings are variable length and libcudf is optimized to minimize costly memory allocations
when operations modify strings. This pattern can make custom UDF logic very difficult to write.

## Strings UDF device library

To make it easier to write UDFs, a device string class (dstring) has been created here to provide
access to the strings inside a libcudf strings column instance and perform specific operations on
individual strings. Once the UDF has been executed on each string, the resulting strings
can be converted back into a libcudf strings column.

## Dependencies
The libcudf strings implementation is available here: https://github.com/rapidsai/cudf.
To build the CLI requires the libcudf and RMM headers as well as libcudf.so and librmm.so.
RMM is available here https://github.com/rapidsai/rmm. 
Follow the [instructions to install cudf](https://github.com/rapidsai/cudf/#conda)
and it should include the RMM dependency automatically.

The Jitify header is also required and available [here](https://github.com/NVIDIA/jitify)

The CUDA tooklkit must be installed and include [NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html)
to build and launch the UDFs.
The code here has only be tested on CUDA 10.1 but may work on earlier (or later) versions.

## Building
You may need to modify the `CONDA_PREFIX` variable in the `build.sh` to point to where
the libcudf and librmm files are located.

Besides cudf and rmm, only the `jitify.hpp` source file is required.
Place it in the same directory as this README.

The CUDA toollkit is expected to be in `/usr/local/cuda`. Modify the `build.sh` if your
installation is in a different directory.

Run the `build.sh` to build the CLI.

## Command Line Interface (CLI)

A CLI is included here to demonstrate executing a UDF on a strings column instance.
The strings column is by created from a text file with new-line delimiters for each string
or from a CSV file. Create the UDF function in a text file to pass to the CLI. 
The output result can be written to a specified file.

The CLI parameters are as follows

| Parameter | Description | Default |
|:---------:| ----------- | ------- |
| -u        | Text file contain UDF kernel function in CUDA/C | (required) |
| -n        | Kernel function name in the UDF file | "udf" |
| -t        | Text or CSV-file for creating strings column | (required) |
| -c        | 0-based column number if CSV file |  0 (first column) |
| -r        | Number of rows to read from file | 0 (entire file) |
| -f        | Output file name/path | default output is stdout |
| -m        | Maximum malloc heap size in MB | 1000 |

Example text
```
aaa bbb cccc ddddd eee
uuv wwww xxxxx yy z
f gg hhh iiii kl
mnopqr stuv wxyz
Aaa bBB ccCC dDDDd
a1234567890
```
Example UDF file (ex1.udf)
```
__global__ void udf( dstring_view* d_in_strs, dstring* d_out_strs, int count )
{
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    if( tid < count )
    {
        dstring str = d_in_strs[tid];
        if( starts_with(str,"a") )
            str = str.insert(0,"+");
        if( str.length() > 10 )
            str = str.substr(0,10);
        d_out_strs[tid] = str;
    }
}
```
Example CLI:
```
$ ./nvs_udf -u ex1.udf -t test.txt
strings count = 6
launch result = 0 [ok]
udf=(0) 0.489572 seconds
0:[+aaa bbb c]
1:[uuv wwww x]
2:[f gg hhh i]
3:[mnopqr stu]
4:[Aaa bBB cc]
5:[+a12345678]
```