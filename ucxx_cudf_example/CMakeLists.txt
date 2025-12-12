cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

# C++ only - no CUDA compiler needed, just CUDA runtime
project(ucxx_cudf_example VERSION 1.0.0 LANGUAGES C CXX)

###################################################################################################
# - Conda environment -----------------------------------------------------------------------------

if(DEFINED ENV{CONDA_PREFIX})
  set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_PREFIX_PATH}")
  set(CONDA_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
  set(CONDA_LINK_DIRS "$ENV{CONDA_PREFIX}/lib")
  # Point CUDAToolkit to conda environment
  set(CUDAToolkit_ROOT "$ENV{CONDA_PREFIX}")
  message(STATUS "Conda environment detected, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")
endif()

###################################################################################################
# - Build type ------------------------------------------------------------------------------------

set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
endif()

###################################################################################################
# - Compiler options ------------------------------------------------------------------------------

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

###################################################################################################
# - Find dependencies -----------------------------------------------------------------------------

# Find CUDA Toolkit (modern CMake way, provides CUDA::cudart target)
find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION}")

# Find cudf (installed via conda)
find_package(cudf REQUIRED)
message(STATUS "Found cudf: ${cudf_VERSION}")

# Find ucxx (installed via conda)
find_package(ucxx REQUIRED)
message(STATUS "Found ucxx: ${ucxx_VERSION}")

# Find rmm (installed via conda)
find_package(rmm REQUIRED)
message(STATUS "Found rmm: ${rmm_VERSION}")

###################################################################################################
# - Build executable ------------------------------------------------------------------------------

add_executable(ucxx_cudf_example src/ucxx_cudf_example.cpp)

# Add conda paths if available
if(CONDA_INCLUDE_DIRS)
  target_include_directories(ucxx_cudf_example PRIVATE "${CONDA_INCLUDE_DIRS}")
endif()

if(CONDA_LINK_DIRS)
  target_link_directories(ucxx_cudf_example PRIVATE "${CONDA_LINK_DIRS}")
endif()

target_link_libraries(ucxx_cudf_example PRIVATE
  cudf::cudf
  ucxx::ucxx
  rmm::rmm
  CUDA::cudart  # CUDA runtime library
)

###################################################################################################
# - Install targets -------------------------------------------------------------------------------

include(GNUInstallDirs)

install(TARGETS ucxx_cudf_example
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
