
import os
import sys
import sysconfig
from distutils.spawn import find_executable
from distutils.sysconfig import get_python_lib

from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import find_packages, setup
from setuptools.extension import Extension

CUDA_HOME = "/usr/local/cuda"
cuda_include_dir = os.path.join(CUDA_HOME, "include")
cuda_lib_dir = os.path.join(CUDA_HOME, "lib64")

CUDF_HOME = "/cudf"
CUDF_ROOT = "/cudf/cpp/build"

extensions = [
    Extension(
        "*",
        sources=["cudfstrings_udf.pyx"],
        include_dirs=[
            os.path.abspath(os.path.join(CUDF_HOME, "cpp/include/cudf")),
            os.path.abspath(os.path.join(CUDF_HOME, "cpp/include")),
            os.path.abspath(os.path.join(CUDF_ROOT, "include")),
            os.path.join(CUDF_ROOT, "_deps/libcudacxx-src/include"),
            os.path.join(CUDF_ROOT, "_deps/dlpack-src/include"),
            os.path.join(
                os.path.dirname(sysconfig.get_path("include")), "libcudf/libcudacxx",
            ),
            os.path.dirname(sysconfig.get_path("include")),
            cuda_include_dir,
        ],
        library_dirs=(
            [get_python_lib(), os.path.join(os.sys.prefix, "lib"), cuda_lib_dir,]
        ),
        libraries=["cudart", "cudf", "nvrtc"],
        language="c++",
        extra_compile_args=["-std=c++17"],
    )
]

setup(
    name="cudfstrings_udf",
    description="cudf strings udf library",
    author="NVIDIA Corporation",
    setup_requires=["cython"],
    ext_modules=cythonize(
        extensions,
        compiler_directives=dict(profile=False, language_level=3, embedsignature=True),
    ),
    zip_safe=False,
)
