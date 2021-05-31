import os
import sys
import sysconfig
import shutil
from distutils.spawn import find_executable
from distutils.sysconfig import get_python_lib

from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import find_packages, setup
from setuptools.extension import Extension

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
print("OS System Lib Path: " + str(os.path.join(os.sys.prefix, "lib")))
print("OS System Include Path: " + str(os.path.dirname(sysconfig.get_path("include"))))

extensions = [
    Extension(
        "*",
        sources=["cudfstrings_udf.pyx"],
        include_dirs=[
            os.path.dirname(sysconfig.get_path("include")),
            os.path.dirname(sysconfig.get_path("include")) + "/libcudf/libcudacxx",
            cuda_include_dir,
            "/usr/local/include",
            "../include",
            "../src",
        ],
        library_dirs=(
            [get_python_lib(), os.path.join(os.sys.prefix, "lib"), cuda_lib_dir, "/usr/local/lib",]
        ),
        libraries=["cudart", "cudf", "nvrtc", "strings_udf"],
        language="c++",
        runtime_library_dirs=[cuda_lib_dir, os.path.join(os.sys.prefix, "lib"), "/usr/local/lib"],
        extra_compile_args=["-std=c++17"],
    )
]

setup(
    name="cudfstrings_udf",
    version="21.08",
    url="https://github.com/rapidsai/rapids-examples.git",
    license="Apache 2.0",
    description="cudf strings udf library",
    author="NVIDIA Corporation",
    setup_requires=["cython"],
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Programming Language :: Python",
    ],
    ext_modules=cythonize(
        extensions,
        compiler_directives=dict(profile=False, language_level=3, embedsignature=True),
    ),
    zip_safe=False,
)