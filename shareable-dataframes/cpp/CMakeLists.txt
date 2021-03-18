cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(RAPIDS-EXAMPLES VERSION 0.19.0 LANGUAGES C CXX CUDA)

# Ensure CUDA is avilable
if(NOT CMAKE_CUDA_COMPILER)
  message(SEND_ERROR "CMake cannot locate a CUDA compiler")
endif(NOT CMAKE_CUDA_COMPILER)

option(BUILD_SHAREABLE_DATAFRAME "Build the strings_udf project" ON)
set(SHAREABLE_DATAFRAME "shareable-dataframes")

###################################################################################################
# - conda environment -----------------------------------------------------------------------------

if("$ENV{CONDA_BUILD}" STREQUAL "1")
    set(CMAKE_PREFIX_PATH "$ENV{BUILD_PREFIX};$ENV{PREFIX};${CMAKE_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{BUILD_PREFIX}/include" "$ENV{PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{BUILD_PREFIX}/lib" "$ENV{PREFIX}/lib")
    message(VERBOSE "CUDF: Conda build detected, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")
elseif(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_PREFIX_PATH}")
    set(CONDA_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
    set(CONDA_LINK_DIRS "$ENV{CONDA_PREFIX}/lib")
    message(VERBOSE "CUDF: Conda environment detected, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")
endif()

#################################################################################################
# - CPM -----------------------------------------------------------------------------------------

set(CPM_DOWNLOAD_VERSION 0.27.2) 
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake")
    file(DOWNLOAD https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()

include(${CPM_DOWNLOAD_LOCATION})

###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations")
endif(CMAKE_COMPILER_IS_GNUCXX)

if(CMAKE_CUDA_COMPILER_VERSION)
  # Compute the version. from  CMAKE_CUDA_COMPILER_VERSION
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR ${CMAKE_CUDA_COMPILER_VERSION})
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR ${CMAKE_CUDA_COMPILER_VERSION})
  set(CUDA_VERSION "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" CACHE STRING "Version of CUDA as computed from nvcc.")
  mark_as_advanced(CUDA_VERSION)
endif()

message(STATUS "CUDA_VERSION_MAJOR: ${CUDA_VERSION_MAJOR}")
message(STATUS "CUDA_VERSION_MINOR: ${CUDA_VERSION_MINOR}")
message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")

# Always set this convenience variable
set(CUDA_VERSION_STRING "${CUDA_VERSION}")

# For now lets keep it simple and only compile CUDA code for specific architectures
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w --expt-extended-lambda --expt-relaxed-constexpr -gencode arch=compute_72,code=sm_72")

####################################################################################################
# - cudf -------------------------------------------------------------------------------------------

# CPMFindPackage(NAME  cudf
#         VERSION         "0.19.0"
#         GIT_REPOSITORY  https://github.com/rapidsai/cudf.git
#         GIT_TAG         branch-0.19
#         GIT_SHALLOW     TRUE
#         SOURCE_SUBDIR   cpp
#         OPTIONS         "BUILD_TESTS OFF"
#                         "BUILD_BENCHMARKS OFF"
#                         "ARROW_STATIC_LIB ON"
#                         "JITIFY_USE_CACHE ON"
#                         "CUDA_STATIC_RUNTIME OFF"
#                         "DISABLE_DEPRECATION_WARNING ON"
#                         "AUTO_DETECT_CUDA_ARCHITECTURES ON"
#     )

######################################################################################################
# - libcudacxx ---------------------------------------------------------------------------------------

function(find_and_configure_libcudacxx VERSION)
    CPMFindPackage(NAME     libcudacxx
        VERSION             ${VERSION}
        GIT_REPOSITORY      https://github.com/NVIDIA/libcudacxx.git
        GIT_TAG             ${VERSION}
        GIT_SHALLOW         TRUE
        DOWNLOAD_ONLY       TRUE
    )
    set(LIBCUDACXX_DIR "${libcudacxx_SOURCE_DIR}" PARENT_SCOPE)
    set(LIBCUDACXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/include" PARENT_SCOPE)
    set(LIBCXX_DIR "${libcudacxx_SOURCE_DIR}/libcxx" PARENT_SCOPE)
    set(LIBCXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/libcxx/include" PARENT_SCOPE)
endfunction()

set(CUDF_MIN_VERSION_libcudacxx 1.4.0)

find_and_configure_libcudacxx(${CUDF_MIN_VERSION_libcudacxx})

######################################################################################################
# - shareable_dataframes -----------------------------------------------------------------------------

if(BUILD_SHAREABLE_DATAFRAME)
  message(STATUS "Building ${BUILD_SHAREABLE_DATAFRAME} example")

  # Create the `shareable_dataframe` shared object
  add_library(shareable_dataframe SHARED
              "src/kernel_wrapper.cu")

  target_include_directories(shareable_dataframe PRIVATE 
                            "${CMAKE_CURRENT_SOURCE_DIR}/include"
                            "${LIBCUDACXX_INCLUDE_DIR}"
                            )

  # Add Conda library paths if specified
  if(CONDA_LINK_DIRS)
    target_link_directories(shareable_dataframe PUBLIC "$<BUILD_INTERFACE:${CONDA_LINK_DIRS}>")
  endif()

  # Add Conda include paths if specified
  if(CONDA_INCLUDE_DIRS)
    target_include_directories(shareable_dataframe PUBLIC "$<BUILD_INTERFACE:${CONDA_INCLUDE_DIRS}>")
  endif()

  target_link_libraries(shareable_dataframe
                        cudf
                        )

endif()



# ###################################################################################################
# # - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${CMAKE_BINARY_DIR}")

###################################################################################################
# - install targets -------------------------------------------------------------------------------

include(GNUInstallDirs)

set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/shareable-dataframes)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME shareable-dataframes)

# install target for shareable_dataframes
install(TARGETS shareable_dataframe
        DESTINATION lib
        EXPORT shareable_dataframes-targets)

# install(DIRECTORY
#             ${CUDF_SOURCE_DIR}/include
#         DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)