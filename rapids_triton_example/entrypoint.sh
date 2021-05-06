#!/bin/bash

# enable conda for this shell
. /root/miniconda3/etc/profile.d/conda.sh

# activate the environment
conda activate rapids-0.20

# exec the cmd/command in this process, making it pid 1
exec "$@"
