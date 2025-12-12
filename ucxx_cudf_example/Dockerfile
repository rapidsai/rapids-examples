# UCXX cuDF Example - Dockerfile
# This Dockerfile creates an environment for building and running
# an example that sends cuDF columns via UCXX communication library.

ARG CUDA_VERSION=12.6.3
ARG LINUX_VERSION=ubuntu22.04

FROM nvidia/cuda:${CUDA_VERSION}-base-${LINUX_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
ARG PARALLEL_LEVEL=8
ENV PARALLEL_LEVEL=${PARALLEL_LEVEL}

# Install basic dependencies
RUN apt-get update -y && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (uses conda-forge by default, no ToS acceptance required)
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

ENV PATH="/opt/conda/bin:${PATH}"
ENV CONDA_PREFIX="/opt/conda"

# Enable conda in bash
SHELL ["/bin/bash", "-c"]

# Copy the project files
COPY . /rapids/ucxx_cudf_example

WORKDIR /rapids/ucxx_cudf_example

# Create the conda environment with libcudf and libucxx from conda
RUN conda env create -f /rapids/ucxx_cudf_example/conda/ucxx_cudf_example.yml

# Configure and build the example
RUN source activate ucxx_cudf_example \
    && cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -S /rapids/ucxx_cudf_example \
        -B /rapids/ucxx_cudf_example/build \
    && cmake --build /rapids/ucxx_cudf_example/build -j${PARALLEL_LEVEL}

# Set up entrypoint to activate conda environment
RUN echo "source activate ucxx_cudf_example" >> ~/.bashrc

SHELL ["/bin/bash", "-l"]
CMD ["/bin/bash", "-l"]
