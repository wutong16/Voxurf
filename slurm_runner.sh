#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
SCRIPT=$3
CONFIG=$4
WORK_DIR=$5
SCENE=$6
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --quotatype=reserved \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    ./${SCRIPT} ${CONFIG} ${WORK_DIR} ${SCENE}
