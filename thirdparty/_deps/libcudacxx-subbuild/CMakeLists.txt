# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.18.5)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(libcudacxx-populate NONE)

include(ExternalProject)
ExternalProject_Add(libcudacxx-populate
                     "UPDATE_DISCONNECTED" "False" "GIT_REPOSITORY" "https://github.com/NVIDIA/libcudacxx.git" "GIT_TAG" "1.4.0" "GIT_SHALLOW" "true"
                    SOURCE_DIR          "/home/jdyer/Development/rapids-examples/thirdparty/_deps/libcudacxx-src"
                    BINARY_DIR          "/home/jdyer/Development/rapids-examples/thirdparty/_deps/libcudacxx-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)


