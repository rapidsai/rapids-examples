FROM nvcr.io/nvidia/tritonserver:21.04-py3

RUN pip install cmake numba

RUN cd /opt/nvidia && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && bash ~/miniconda.sh -b
ENV PATH="/root/miniconda3/bin:$PATH"
RUN conda init bash \ 
     && . ~/.bashrc \ 
     && conda create -n rapids -c rapidsai-nightly -c pytorch -c nvidia  -c conda-forge cudf pytorch=1.7.1 transformers python=3.8 cudatoolkit=11.0


COPY ./entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
