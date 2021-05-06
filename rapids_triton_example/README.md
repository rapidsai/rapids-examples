# Triton + Rapids Example

## Triton
Triton Inference Server simplifies the deployment of AI models at scale in production. It lets teams deploy trained AI models from any framework (TensorFlow, NVIDIA® TensorRT, PyTorch, ONNX Runtime, or custom) from local storage or cloud platform on any GPU- or CPU-based infrastructure (cloud, data center, or edge) and deploy them on the cloud. 

Check out the Triton documentation at [link](https://github.com/triton-inference-server/server/blob/r21.04/README.md#documentation)

## Using Rapids and Triton together:

We use Triton's [python backend](https://github.com/triton-inference-server/python_backend), which allows you to server Python models that can execute arbitrary python code. 

Here we showcase a simple example of using Rapids with Triton.

## Build 

[build.sh](build.sh) creates a Triton+Rapids docker container which you can use to deploy your rapids code with Triton.  

```bash
bash build.sh
```


### Model Code:
The example model here does tokenization of string logs into numerical vectors using `cuDF's subwordTokenizer.`  

Python model code is present in [models/rapids_tokenizer/1/model.py](models/rapids_tokenizer/1/model.py)

The model configuration is defined in [models/rapids_tokenizer/config.pbtxt](models/rapids_tokenizer/config.pbtxt)


## Serving:
Triton inference server is started using [start_server.sh](start_server.sh). 

```bash
bash start_server.sh
```


### Client Code:
The client logic to interact with the served Triton model is present in [example_client.ipynb](example_client.ipynb). 
