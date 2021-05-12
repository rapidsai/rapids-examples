FROM nvcr.io/nvidia/tritonserver:21.04-py3

RUN pip install cmake numba

RUN cd /opt/nvidia && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && bash ~/miniconda.sh -b
ENV PATH="/root/miniconda3/bin:$PATH"
RUN conda init bash \ 
     && . ~/.bashrc \ 
     && conda create -n rapids-0.20  -c rapidsai-nightly -c nvidia -c conda-forge -c defaults rapids=0.20 python=3.8 cudatoolkit=11.2 


COPY ./entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
