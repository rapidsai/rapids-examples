# MiNiFi cuDF

This repository is aimed to show an example of how to use Apache MiNiFi in conjunction with RAPIDS cuDF. It demonstrates how to properly setup MiNiFi to work with Python native processors and gives a small example of using those Python native processors to interact with RAPIDS cuDF.

## Building Docker Image

`docker build -t rapids-examples/minifi_cudf --build-arg PARALLEL_LEVEL=10 --build-arg CONDA_ENV_NAME=minifi_cudf .`

| Docker ARG      | Description         | Required?         |
| :---            | :----               | :---              |
| PARALLEL_LEVEL  | # cores used for building MiNiFi source code               | NO, default = 10  |
| CONDA_ENV_NAME  | Name of Conda environment that will be created              | NO, default = `minifi_cudf`  |
