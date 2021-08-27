rm -rf models/.ipynb_checkpoints/

docker run --rm  --gpus='all' --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/models:/models -it rapidstriton tritonserver --model-repository /models
