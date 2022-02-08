#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
GPUS=($(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n')) # array of gpus

OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OMP_NUM_THREADS

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --standalone --nproc_per_node=${#GPUS[@]} --nnodes=1 \
train.py "${@:1}" --training.distributed=true