# Shareable Dataframes
## Overview
Real world development teams are quite varied. Team members often have different knowledge in toolsets and business procedures. This example's intention is the solve the challenge of team members with different software skill sets by allowing each team member to use their primary development language while contributing to solving a common business requirement with Nvidia RAPIDS.

This example demonstrates how to share cudf dataframes between Python and custom CUDA kernels. This is useful for performing custom CUDA-accelerated business logic on cuDF dataframes and handling certain tasks in Python and others in CUDA.

Dataframes that are created in Python cuDF are already present in GPU memory and accessible to CUDA code. This makes it straightforward to write a CUDA kernel to work with a dataframe columns. In fact this is how libcudf processes dataframes in CUDA kernels; the only difference in this example is that we invoke CUDA kernels that exist outside the cuDF code base. The term User Defined Function (UDF) could be loosely used to describe what this example is demonstrating.

This example provides a Cython `kernel_wrapper` implementation to make sharing the dataframes between Python and our custom CUDA kernel easier. This wrapper allows Python users to seamlessly invoke those CUDA kernels with a single function call and also provides a clear place to implement the C++ "glue code".

The example CUDA kernel accepts a data column (PRCP) containing rainfall values stored as 1/10th of a mm and converts those values to inches. The dataframe is read from a local CSV file using Python. Python then invokes the CUDA mm->inches conversion kernel via the Cython `kernel_wrapper`, passing it the dataframe object. The converted data can then be accessed from Python, e.g. using `df.head()`.

This is similar to an existing [weather notebook](https://github.com/rapidsai/notebooks-contrib/blob/branch-0.19/intermediate_notebooks/examples/weather.ipynb), which provides a reference for understanding the implementation. 

## Building (Inside Docker container)

1. Compile C++ `kernel_wrapper` code and CUDA kernels
    - ```cmake -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DCMAKE_CUDA_ARCHITECTURES= -S /rapids/rapids-examples/shareable-dataframes/cpp -B /rapids/rapids-examples/shareable-dataframes/cpp/build```
    - ```cmake --build /rapids/rapids-examples/shareable-dataframes/cpp/build -j${PARALLEL_LEVEL} -v```
    - ```cmake --install /rapids/rapids-examples/shareable-dataframes/cpp/build -v```
2. Create the `shareable_dataframes` conda environment for the python code. 
    - ```conda env create -f /rapids/rapids-examples/shareable-dataframes/conda/shareable_dataframes.yml --name shareable_dataframes```
3. Activate the `shareable_dataframes` conda environment ```conda activate shareable_dataframes```
4. Build the cython kernel_wrapper code, this will also link against the previously compiled C++ code.
    - ```cd /rapids/rapids-examples/shareable-dataframes/python```
    - ```python setup.py build_ext --inplace```
    - ```python setup.py install```
5. Download weather data. A convenience Python script has been provided here to make that easier for you. By default it will download years 2010-2020 weather data. That data is about 300MB per file so if you need to download less files you can change that in the script. The data will be downloaded to ./data/weather.
    - ```python /rapids/rapids-examples/shareable-dataframes/data/download_data.py```
6. Run the Python example script. It expects an input of a single Weather year file. - EX: ```python /rapids/rapids-examples/shareable-dataframes/python/python_kernel_wrapper.py /rapids/rapids-examples/shareable-dataframes/python/data/weather/2010.csv.gz```

CUDA Kernel with existing business logic:
``` cpp
#include <stdio.h>
#include <cudf/column/column_device_view.cuh> // cuDF component
#include <cudf/table/table_device_view.cuh> // cuDF component

static constexpr float mm_to_inches = 0.0393701;

// cudf::mutable_column_device_view used in place of device memory buffer
__global__ void kernel_tenth_mm_to_inches(cudf::mutable_column_device_view column)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < column.size()) {
        // Access and set cuDF element
        column.element<int64_t>(i) = column.element<int64_t>(i) * (1/10) * mm_to_inches;
    }
}
```

Cython wrapper "glue":
```python
import cudf
from cudf._lib.cpp.table.table_view cimport mutable_table_view
from cudf._lib.table cimport Table
from libcpp.string cimport string

cdef extern from "src/kernel_wrapper.hpp":
    cdef cppclass C_CudfWrapper "CudfWrapper":
        C_CudfWrapper(mutable_table_view tbl)
        void tenth_mm_to_inches(int column_index)

cdef class CudfWrapper:
    cdef C_CudfWrapper* gdf

    def __cinit__(self, Table t):
        self.gdf = new C_CudfWrapper(t.mutable_view())

    def tenth_mm_to_inches(self, col_index):
        self.gdf.tenth_mm_to_inches(col_index)
```

Python logic using CUDA kernel:
``` python
import cudf
import cudfkernel  # Cython bindings to execute existing CUDA Kernels

# CSV reader options; names of columns from weather data csv file
column_names = [
    "station_id",
    "date",
    "type",
    "val",
    "m_flag",
    "q_flag",
    "s_flag",
    "obs_time",
]
usecols = column_names[0:4]

# Create weather dataframe
weather_df = cudf.read_csv(
    weather_file_path, names=column_names, usecols=usecols
)

# There are 5 possible recording types. PRCP, SNOW, SNWD, TMAX, TMIN
# Rainfall is stored as 1/10ths of MM.
rainfall_df = weather_df[weather_df["type"] == "PRCP"]

# Wrap the rainfall_df for CUDA to consume
rainfall_kernel = cudfkernel.CudfWrapper(rainfall_df)  

# Run the custom Kernel on the specified Dataframe Columns, index 4 is the "val" column
rainfall_kernel.tenth_mm_to_inches(4)

# Shows head() after rainfall totals have been altered
print(rainfall_df.head())

```