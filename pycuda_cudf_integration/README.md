# Using PyCuda and Cudf
## Overview
Sometimes, a user may wish to run custom device code to do numerical operations on a `cuDF` column. 

One way to do this is through custom cython bindings which can require significant setup work as shown [here](https://github.com/rapidsai/rapids-examples/tree/main/shareable-dataframes).

With `PyCUDA`,  the custom CUDA kernels can be run directly using its` sourcModule` to modify the `cuDF`  dataframe.   

`PyCUDA` has limitations, especially around running host-side code, which can are noted in detail in the limitations section.  

## PyCUDA Interactions
In order to write custom cuda kernel in `pycuda` we make use of `SourceModule`. After constructing the kernel, we can retrieve the function and store it in a variable like below.
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
The interactions between `cudf` and `pycuda` is dependent upon the implementation of the `__cuda_array_interface__`, which serves as a contract to tell the world of how to convert the underlying data between array-like data structures, without requiring a copy.
```python
import cudf
import cupy as cp

df = cudf.DataFrame({'col': [i for i in range(200000)]})
length = cp.int32(len(df['col']))

func(df['col'], length, block=(256,1,1), grid=(4096,))
```

## PyCuda Limitations
At its core, `pycuda` is meant for writing CUDA kernels that operate on fixed length columns. One interesting fact about `pycuda` is that it allows including external libraries, such as `thrust` or `libcudf`. However, the issue becomes that pycuda allows device-based code in external libraries such as `thrust`, but does not support running host code.

## Working with external CUDA libraries
To showcase how this can be done, we will be using a simple example of how we can fill a `cudf` column with random numbers.

```python
import pycuda.autoprimaryctx
from pycuda.compiler import SourceModule
import cudf
import cupy as cp

df = cudf.DataFrame({'col': cp.zeros(200000)})
length = cp.int32(len(df['col']))
```
We can construct the sample kernel like below,
```python
mod = SourceModule('''
#include <thrust/random.h>

extern "C" {

    __global__ void random_column(double* a, int N)
    {
      thrust::default_random_engine rng;
      
      int stride = blockDim.x * gridDim.x;
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
        if (i < N) {
          rng.discard(i);
          thrust::uniform_real_distribution<float> rand01(0,1);
          double r = rand01(rng);
          a[i] = r;
        }
      }
    }
}
''', no_extern_c=True)

func = mod.get_function('random_column')
func(df['col'], length, block=(256,1,1), grid=(4096,))
```
Here we use `thrust` to generate random numbers to fill in the cudf. In some cases, the CUDA library you want to use may be stored in other locations (such as when you install `libcudf` in a conda environment), in that case, we must include the directories in `SourceModule` like below.
```python
cuda_includes = ['/raid/sding/miniconda3/envs/cudf_dev/include', '/raid/sding/miniconda3/envs/cudf_dev/include/libcudf/libcudacxx']

mod = SourceModule('''
#include <cudf/column/column_device_view.cuh>

extern "C" {

    __global__ void random_column(double* a, int N)
    {
        return;
    }
}
''', no_extern_c=True, include_dirs=cuda_includes, options=['--expt-relaxed-constexpr'])
```
Two key things to note, for `libcudf` to work, it requires both the `include/` inside the conda environment in which you installed `cudf` from source and also `libcudacxx/` inside `libcudf`. 

If you want include additional `nvcc` flags, you can add them to the list given to `options`.

## When to use what?
After reading the above, a question that may come to mind is when should you use `Pycuda`, `Cython` or any other alternatives that allow interfacing between `Python` and low-level `C` code? `Pycuda` is likely best used when the developer is looking for an easy and low overhead way of interfacing with a `CUDA` kernel for the purpose of accelerating an operation on their fixed length column data, as `Pycuda` allows the developer to access the power of GPU with little overhead. On the other hand, if the developer is okay with a large amount of `Cython` overhead and requires the ability to execute host-level code, than building `Cython` bindings will likely be the more appropriate option.