ARG CUDA_VERSION=11.1.1
ARG LINUX_VERSION=ubuntu18.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${LINUX_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
ARG PARALLEL_LEVEL=10
ENV PARALLEL_LEVEL=10
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install dev dependencies and tools
RUN GCC_VERSION=$(bash -c '\
CUDA_VERSION=$(nvcc --version | head -n4 | tail -n1 | cut -d" " -f5 | cut -d"," -f1); \
CUDA_VERSION_MAJOR=$(echo $CUDA_VERSION | tr -d '.' | cut -c 1-2); \
CUDA_VERSION_MINOR=$(echo $CUDA_VERSION | tr -d '.' | cut -c 3); \
  if [[ "$CUDA_VERSION_MAJOR" == 9 ]]; then echo "7"; \
  elif [[ "$CUDA_VERSION_MAJOR" == 10 ]]; then echo "8"; \
  elif [[ "$CUDA_VERSION_MAJOR" == 11 ]]; then echo "9"; \
  else echo "10"; \
  fi') \
&& apt update -y \
&& apt install -y software-properties-common \
&& add-apt-repository -y ppa:git-core/ppa \
&& add-apt-repository -y ppa:ubuntu-toolchain-r/test \
&& apt update -y \
&& apt install -y \
   build-essential git wget vim\
   gcc-${GCC_VERSION} g++-${GCC_VERSION} \
   # CMake dependencies
   curl libssl-dev libcurl4-openssl-dev zlib1g-dev \
   # cuDF dependencies
   libboost-filesystem-dev \
&& apt autoremove -y \
&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
&& update-alternatives \
   --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100 \
# Set gcc-${GCC_VERSION} as the default gcc
&& update-alternatives --set gcc /usr/bin/gcc-${GCC_VERSION}

ARG CMAKE_VERSION=3.18.5

# Install miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /conda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

ENV PATH=${PATH}:/conda/bin
# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

# checkout the repo inside the container making it more portable
RUN mkdir -p /rapids \
 && git clone https://github.com/jdye64/rapids-examples.git /rapids/rapids-examples \
 && cd /rapids/rapids-examples \
 && git checkout rapids-devel-base

# Create shareable_dataframes conda environment
RUN conda env create -f /rapids/rapids-examples/shareable-dataframes/conda/shareable_dataframes.yml --name shareable_dataframes

# Configure
RUN source activate shareable_dataframes \
    && cmake \
    -DBUILD_TESTS=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="ALL" \
    -S /rapids/rapids-examples/shareable-dataframes/cpp \
    -B /rapids/rapids-examples/shareable-dataframes/cpp/build

# Build
RUN source activate shareable_dataframes \
    && cmake --build /rapids/rapids-examples/shareable-dataframes/cpp/build -j${PARALLEL_LEVEL} -v

# Install (default install to `/usr/local/{include|lib}` prefix)
RUN source activate shareable_dataframes \
    && cmake --install /rapids/rapids-examples/shareable-dataframes/cpp/build -v

# Since the default install location is /usr/local/lib. Lets copy to the anaconda env to make things easier for Python development
RUN cp /usr/local/lib/libshareable_dataframe.so /root/miniconda3/envs/shareable_dataframes/lib/.

SHELL ["/bin/bash", "-l"]
CMD ["/bin/bash", "-l"]