# Rapids Examples

### Assumptions
1. [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install) is installed.
2. [CUDA](https://developer.nvidia.com/cuda-downloads) 10.0 > is installed and on the PATH.

## Examples List
| Example | Description                                     |
|:-------:| :-----------------------------------------------|
[python-kernel-wrapper](./python-kernel-wrapper) | Demonstrates processing python cudf dataframes in a cuda kernel
[pycuda\_cudf\_integration](./pycuda_cudf_integration) | Demonstrates processing python cudf dataframes using `pycuda`
[tfidf-benchmark](./tfidf-benchmark) | Benchmarks NLP text processing pipeline in cuML + Dask vs. Apache Spark
[rapids_triton_example](./rapids_triton_example) |  Example of using RAPIDS+Pytorch with Nvidia Triton.
