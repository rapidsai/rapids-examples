import os
import shutil
import sysconfig
from distutils.sysconfig import get_python_lib

import numpy
from Cython.Distutils import build_ext
from setuptools import Extension, find_packages, setup

# Locate CUDA_HOME
CUDA_HOME = os.environ.get("CUDA_HOME", False)
if not CUDA_HOME:
    path_to_cuda_gdb = shutil.which("cuda-gdb")
    if path_to_cuda_gdb is None:
        raise OSError(
            "Could not locate CUDA. "
            "Please set the environment variable "
            "CUDA_HOME to the path to the CUDA installation "
            "and try again."
        )
    CUDA_HOME = os.path.dirname(os.path.dirname(path_to_cuda_gdb))

cuda_include_dir = os.path.join(CUDA_HOME, "include")
cuda_lib_dir = os.path.join(CUDA_HOME, "lib64")

print("CUDA Include Dir: " + cuda_include_dir)
print("CUDA Library Dir: " + cuda_lib_dir)

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


ext = Extension(
    "cudfkernel",
    sources=["kernel.pyx"],
    library_dirs=[
        cuda_lib_dir,
        get_python_lib(),
        os.path.join(os.sys.prefix, "lib"),
        "/test/cpp/build"
    ],
    libraries=["cudf", "cudart", "shareable_dataframe"],
    language="c++",
    runtime_library_dirs=[cuda_lib_dir, "/test/cpp/build"],
    include_dirs=[
        os.path.dirname(sysconfig.get_path("include")),
        numpy_include,
        cuda_include_dir,
        "/test/cpp/include"
    ],
)

setup(
    name="cudfkernel",
    version="0.1",
    url="https://github.com/rapidsai/cudf",
    author="NVIDIA Corporation",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Programming Language :: Python",
    ],
    ext_modules=[ext],
    packages=find_packages(include=["cudf", "cudf.*"]),
    zip_safe=False,
)