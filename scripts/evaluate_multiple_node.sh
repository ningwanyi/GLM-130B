#!/bin/bash

NUM_WORKERS=16
NUM_GPUS_PER_WORKER=8
HOST_FILE_PATH="<your hostfile>"
OPTIONS_NCCL="NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 CUDA_LAUNCH_BLOCKING=0"

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

source "${main_dir}/configs/model_glm_130b.sh"

DATA_PATH="<your evaluation dataset base directory>"

ARGS="${main_dir}/evaluate.py \
       --mode inference \
       --data-path $DATA_PATH \
       --task $* \
       $MODEL_ARGS"

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
EXP_NAME=${TIMESTAMP}

mkdir -p logs

run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} ${ARGS}"
eval ${run_cmd} 2>&1 | tee logs/${EXP_NAME}.log
