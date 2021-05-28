# Using PyCuda and Cudf
## Overview
Sometimes, a user may wish to run custom device code to do numerical operations on a `cuDF` column. 

One way to do this is through custom cython bindings which can require significant setup work as shown [here](https://github.com/rapidsai/rapids-examples/tree/main/shareable-dataframes).

With `PyCUDA`,  the custom CUDA kernels can be run directly using its` sourcModule` to modify the `cuDF`  dataframe.   

`PyCUDA` has limitations, especially around running host-side code, which can are noted in detail in the limitations section.  

## PyCuda interactions
In order to write custom cuda kernel in `pycuda` we make use of `SourceModule` provided to us by the library. After constructing the kernel, we can retreive the function and store it in a variable like below.
```python
import pycuda.autoprimaryctx
from pycuda.compiler import SourceModule
mod = SourceModule("""
    __global__ void doublify(int64_t *a, int N)
    {
      int stride = blockDim.x * gridDim.x;
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
        if (i < N) {
          a[i] *= 2;
        }
      }
    }
    """)
func = mod.get_function("doublify")
```
The interactions between `cudf` and `pycuda` is dependent upon the implementation of `cudf` columns as `cupy` arrays, which has the `__cuda_array_interface__` property implemented. Allowing us to pass on information regarding the column directly to `pycuda`, by passing it as a parameter like below.
```python
import cudf
import cupy as cp

df = cudf.DataFrame({'col': [i for i in range(200000)]})
length = cp.int32(len(df['col']))

func(df['col'], size, block=(256,1,1), grid=(4096,))
```

## PyCuda Limitations
At its core, `pycuda` is meant for writing CUDA kernels that operate on fixed length columns. This limits its usage (at least for now) to numeric columns only. Furthermore, one interesting fact about `pycuda` is that it allows including external libraries, such as `thrust` or `libcudf`. However, the issue becomes that the majority of these external libraries are meant to run on host code, which is not possible via `SourceModule`.
