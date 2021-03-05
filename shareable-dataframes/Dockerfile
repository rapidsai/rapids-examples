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
&& add-apt-repository ppa:deadsnakes/ppa \
&& apt update -y \
&& apt install -y \
   build-essential git wget \
   gcc-${GCC_VERSION} g++-${GCC_VERSION} \
   # CMake dependencies
   curl libssl-dev libcurl4-openssl-dev zlib1g-dev \
   # cuDF dependencies
   libboost-filesystem-dev \
   # # Install Python 3.8
   # python3.8 \
&& apt autoremove -y \
&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
&& update-alternatives \
   --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100 \
# Set gcc-${GCC_VERSION} as the default gcc
&& update-alternatives --set gcc /usr/bin/gcc-${GCC_VERSION}

ARG CMAKE_VERSION=3.18.5

# # Alias for python->python3.8
# RUN echo 'alias python="python3.8"' >> ~/.bashrc

# Install miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda init bash

# Install CMake
RUN cd /tmp \
 && curl -fsSLO --compressed "https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION.tar.gz" -o /tmp/cmake-$CMAKE_VERSION.tar.gz \
 && tar -xvzf /tmp/cmake-$CMAKE_VERSION.tar.gz && cd /tmp/cmake-$CMAKE_VERSION \
 && /tmp/cmake-$CMAKE_VERSION/bootstrap \
    --system-curl \
    --parallel=${PARALLEL_LEVEL} \
 && make install -j${PARALLEL_LEVEL} \
 && cd /tmp && rm -rf /tmp/cmake-$CMAKE_VERSION*

# checkout the repo inside the container making it more portable
RUN mkdir -p /rapids \
 && git clone https://github.com/jdye64/rapids-examples.git /rapids/rapids-examples

# Configure
RUN cmake \
    -DBUILD_TESTS=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES= \
    -S /rapids/rapids-examples/shareable-dataframes/cpp \
    -B /rapids/rapids-examples/shareable-dataframes/cpp/build

# Build
RUN cmake --build /rapids/rapids-examples/shareable-dataframes/cpp/build -j${PARALLEL_LEVEL} -v

# Install (default install to `/usr/local/{include|lib}` prefix)
RUN cmake --install /rapids/rapids-examples/shareable-dataframes/cpp/build -v

# Since the default install location is /usr/local/lib. Lets copy to the anaconda env to make things easier for Python development
RUN cp /usr/local/lib/libshareable_dataframe.so /root/miniconda3/envs/shareable_dataframes/lib/.

SHELL ["/bin/bash", "-l"]
CMD ["/bin/bash", "-l"]
