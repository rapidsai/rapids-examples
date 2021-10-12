FROM nvcr.io/nvidia/tritonserver:21.09-py3

SHELL ["/bin/bash", "--login", "-c"]

RUN cd /opt/nvidia && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && bash ~/miniconda.sh -b
ENV PATH="/root/miniconda3/bin:$PATH"
RUN conda init bash \ 
     && . ~/.bashrc \ 
     && conda create -n rapids -c rapidsai-nightly -c pytorch -c nvidia  -c conda-forge cudf pytorch transformers python=3.8 cudatoolkit=11.0 conda-pack

RUN conda init bash

SHELL ["conda", "run", "-n", "rapids", "/bin/bash", "-c"]

RUN export PYTHONNOUSERSITE=True & conda-pack -o /root/miniconda3/envs/rapids/rapids.tar.gz
COPY ./entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
