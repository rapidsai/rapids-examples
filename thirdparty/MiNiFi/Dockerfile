ARG CUDA_VERSION=11.2.2
ARG LINUX_VERSION=ubuntu18.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${LINUX_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
ARG PARALLEL_LEVEL=10
ENV PARALLEL_LEVEL=10
ARG CONDA_ENV_NAME="minifi_cudf"
ENV CONDA_ENV_NAME="minifi_cudf"
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

ENV ROOT_DIR="/minifi"

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
   # Apache MiNiFi build dependencies
   bison \
   flex \
   patch \
   uuid-dev \
   uuid \
   libboost-all-dev \
   libssl-dev \
   libbz2-dev \
   liblzma-dev \
   zlib1g-dev \
   libpython3-dev \
   libpcap-dev \
   autoconf \
   automake \
   libtool \
   libcurl4-openssl-dev \
&& apt autoremove -y \
&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
&& update-alternatives \
   --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 100 \
# Set gcc-${GCC_VERSION} as the default gcc
&& update-alternatives --set gcc /usr/bin/gcc-${GCC_VERSION}

# Install miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /conda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

ENV PATH=${PATH}:/conda/bin
# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

# Add runtime components to the container
ADD conda ${ROOT_DIR}/conda
ADD conf ${ROOT_DIR}/conf
ADD data  ${ROOT_DIR}/data
ADD minifi-python ${ROOT_DIR}/minifi-python

# Create minifi_cudf conda environment
RUN conda env create -f ${ROOT_DIR}/conda/minifi_cudf.yml --name ${CONDA_ENV_NAME}

# Clone minifi locally to be built
RUN git clone https://github.com/apache/nifi-minifi-cpp.git ${ROOT_DIR}/minifi-github

# Configure
RUN source activate ${CONDA_ENV_NAME} \
    && mkdir -p ${ROOT_DIR}/minifi-github/build \
    && cmake \
    -DENABLE_LIBRDKAFKA=ON \
    -DSKIP_TESTS=true \
    -DASAN_BUILD=OFF \
    -DUSE_SHARED_LIBS=ON \
    -DPORTABLE=ON \
    -DBUILD_ROCKSDB=ON \
    -DFAIL_ON_WARNINGS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=/conda/envs/${CONDA_ENV_NAME}/bin/python \
    -DCMAKE_INSTALL_PREFIX=/conda/envs/${CONDA_ENV_NAME} \
    -S ${ROOT_DIR}/minifi-github \
    -B ${ROOT_DIR}/minifi-github/build

# Build
RUN source activate ${CONDA_ENV_NAME} \
    && cmake --build ${ROOT_DIR}/minifi-github/build -j${PARALLEL_LEVEL} -v

# Install (default install to `/usr/local/{include|lib}` prefix)
RUN source activate ${CONDA_ENV_NAME} \
    && cmake --install ${ROOT_DIR}/minifi-github/build -v

# Move the minifi executable the the appropriate location
RUN mkdir ${ROOT_DIR}/bin/ \
    && mv ${ROOT_DIR}/minifi-github/build/main/minifi ${ROOT_DIR}/bin/

SHELL ["/bin/bash", "-l"]
CMD ["/bin/bash", "-l"]
