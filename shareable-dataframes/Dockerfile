FROM rapidsai/rapidsai-dev-nightly:0.18-cuda11.0-devel-ubuntu20.04-py3.8

RUN apt update && apt install -y libboost-all-dev

RUN mkdir -p /rapids/rapids-examples
ADD . /rapids/rapids-examples/shareable-dataframes
RUN mkdir -p /rapids/rapids-examples/shareable-dataframes/cpp/build
WORKDIR /rapids/rapids-examples/shareable-dataframes/cpp/build

# Default install to `/usr/local/include|lib}` prefix
RUN cmake .. && make -j5 && make install
