# cuBERT-topic-modelling

Leveraging BERT, TF-IDF and NVIDIA RAPIDS to create easily interpretable topics.

## Overview

Currently, [BERTopic](https://github.com/MaartenGr/BERTopic) library in general utilizes GPUs as part of the SenteneTransformer package, which is only used during the encoding process. We can accelerate other bits of the pipeline such as dimension reduction (UMAP), Topic Creation (Count/TF-IDF Vectorization), Clustering (HDBSAN) and Topic Reduction using the RAPIDS ecosystem that has a similar API as Pandas, sckit-learn but leverages GPUs for the end-to-end pipeline.

<img src="cuBERT_topic_modelling/cuBERTopic.jpg" width="300" height="400">

## Installation

`cuBERTopic` runs on `cudf` and `cuml` which can be installed using instructions[here](https://rapids.ai/start.html) and `cupy` which can be installed using instructions [here](https://docs.cupy.dev/en/stable/install.html)

After installing the dependencies, clone the repository and you can now use `cuBERTopic`!

## Quick Start

An [example](berttopic_example.ipynb) notebook is provided, which goes through the installation, as well as comparing response using [BERTopic](https://github.com/MaartenGr/BERTopic).

## Acknowledgement

Our work has been inspired from the [BERTopic library](https://github.com/MaartenGr/BERTopic) and Maarten Grootendorst's [blog](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6) on how to use BERT to create your own topic model.


