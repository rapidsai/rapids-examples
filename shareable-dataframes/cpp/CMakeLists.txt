cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(RAPIDS-EXAMPLES VERSION 0.19.0 LANGUAGES C CXX CUDA)

# Ensure CUDA is avilable
if(NOT CMAKE_CUDA_COMPILER)
  message(SEND_ERROR "CMake cannot locate a CUDA compiler")
endif(NOT CMAKE_CUDA_COMPILER)

option(BUILD_SHAREABLE_DATAFRAME "Build the strings_udf project" ON)
set(SHAREABLE_DATAFRAME "shareable-dataframes")

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
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w --expt-extended-lambda --expt-relaxed-constexpr -gencode arch=compute_70,code=sm_70")

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

####################################################################################################
# - SPDLOG -----------------------------------------------------------------------------------------

CPMAddPackage(
    NAME spdlog
    GITHUB_REPOSITORY gabime/spdlog
    VERSION 1.7.0)

####################################################################################################
# - cxxopts ----------------------------------------------------------------------------------------

CPMAddPackage(
  NAME cxxopts
  GITHUB_REPOSITORY jarro2783/cxxopts
  VERSION 2.2.0
  OPTIONS
    "CXXOPTS_BUILD_EXAMPLES Off"
    "CXXOPTS_BUILD_TESTS Off"
)

####################################################################################################
# - cudf -------------------------------------------------------------------------------------------

# XXX - Waiting on PR https://github.com/rapidsai/cudf/pull/7107 before this can be added
function(find_and_configure_cudf VERSION)

    CPMAddPackage(NAME  cudf
        VERSION         ${VERSION}
        #GIT_REPOSITORY  https://github.com/rapidsai/cudf.git
        #GIT_TAG         branch-0.19-latest
        GIT_REPOSITORY  https://github.com/trxcllnt/cudf.git
        GIT_TAG         fix/cmake-exports
        GIT_SHALLOW     TRUE
        SOURCE_SUBDIR   cpp
        OPTIONS         "BUILD_TESTS OFF"
                        "BUILD_BENCHMARKS OFF"
                        "ARROW_STATIC_LIB ON"
                        "JITIFY_USE_CACHE ON"
                        "CUDA_STATIC_RUNTIME ON"
                        "DISABLE_DEPRECATION_WARNING ON"
                        "AUTO_DETECT_CUDA_ARCHITECTURES ON"
    )

endfunction()

find_and_configure_cudf("0.19.0")

# add_target(${PROJECT_NAME} SHARED "${src_files}")
# target_link_libraries(${PROJECT_NAME} rmm::rmm cudf::cudf)

######################################################################################################
# - shareable_dataframes -----------------------------------------------------------------------------

if(BUILD_SHAREABLE_DATAFRAME)
  message(STATUS "Building ${BUILD_SHAREABLE_DATAFRAME} example")

  # XXX - Understand if RMM will be included as part of the upcoming CPMAddPackage work for CUDF? If not will need to hack a way to add that here ourselves. Hopefully it is however.
  add_executable(shareable_dataframe "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel.cu"
                                     "${CMAKE_CURRENT_SOURCE_DIR}/src/kernel_wrapper.cu"
                             )

  # XXX - Remove this hardcoded path evilness, only needed right now since CPM is not complete.
  message(STATUS "Include directory I need: ${CMAKE_CURRENT_SOURCE_DIR}/${SHAREABLE_DATAFRAME}/include")
  target_include_directories(shareable_dataframe PRIVATE 
                            "${CMAKE_CURRENT_SOURCE_DIR}/${SHAREABLE_DATAFRAME}/include"
                            "/home/jdyer/anaconda3/envs/cudf_dev/include"
                            "/home/jdyer/anaconda3/envs/cudf_dev/include/libcudf/libcudacxx"
                            "/home/jdyer/Development/cudf/thirdparty/jitify")

  # XXX - Remove this hardcoded path evilness, only needed right now since CPM is not complete.
  target_link_directories(shareable_dataframe PRIVATE "/home/jdyer/anaconda3/envs/cudf_dev/lib")
  target_link_libraries(shareable_dataframe
                        rmm::rmm
                        cudf::cudf
                        dl
                        cuda
                        nvrtc)

endif()



# ###################################################################################################
# # - library paths ---------------------------------------------------------------------------------

# link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
#                  "${CMAKE_BINARY_DIR}/lib"
#                  "${CMAKE_BINARY_DIR}")

# if(CONDA_LINK_DIRS)
#     link_directories("${CONDA_LINK_DIRS}")
# endif(CONDA_LINK_DIRS)

# ###################################################################################################
# # - library targets -------------------------------------------------------------------------------
# add_library(kernel "src/kernel.cu" "src/kernel_wrapper.cu")

# # ###################################################################################################
# # # - link libraries --------------------------------------------------------------------------------

# # target_link_libraries("${CUDF_PROXY_NAME}" "${CUDF_NAMESPACE}::${CUDF_BASE_NAME}" "${CUDF_MODULES}")