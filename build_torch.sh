#!/bin/bash

##
# Script for building Pytorch from source
# Tharindu Patabandi <tharindu@protonmail.com>
##

export USE_CUDA=0
export USE_ROCM=0
export USE_XPU=0

## pull pytorch code and submodules
# git clone https://github.com/pytorch/pytorch
# cd pytorch
## if you are updating an existing checkout
# git submodule sync
# git submodule update --init --recursive

pip install --group dev

# arch=$(uname -m)

make triton

export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-"$(dirname "$(which conda)")/../"}:${CMAKE_PREFIX_PATH}"

python -m pip install --no-build-isolation -v -e .
