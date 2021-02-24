FROM rapidsai/rapidsai-dev-nightly:0.18-cuda11.0-devel-ubuntu20.04-py3.8

RUN mkdir -p /rapids/rapids-examples
ADD . /rapids/rapids-examples/shareable-dataframes
WORKDIR /rapids/rapids-examples/shareable-dataframes/cpp
RUN mkdir build
WORKDIR build
RUN cmake .. && make -j10
