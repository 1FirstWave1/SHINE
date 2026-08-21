#!/bin/bash

set -euxo pipefail

# Run this same script once on every node. torchrun forms one DDP job across
# the nodes; this training path does not need a Ray cluster.
NAME=8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150 # 4layer
CONFIG_NAME="Qwen3-8B"       
SOURCE=grouptransmla
TRAIN_BATCH_SIZE=1
TEST_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
USE_GRADIENT_CHECKPOINT=True
RESUME_GLOBAL_STEP=-1   # -1: don't resume,   int: resume from global steps,  latest: resume from latest
LEARNING_RATE=5e-5
CONVERSATION_MAX_LEN=1150 # 1160   # Extra base len: 0 Extra chat len per turn: 11
CONTEXT_MAX_LEN=$((CONVERSATION_MAX_LEN - 9)) # $((CONVERSATION_MAX_LEN - 10))
TYPE=transformer
NUM_LAYERS=4
WARMUP_STEPS=200
METHOD=rl
LORA_R=8
METALORA_R=128

# Multi-node settings from the ModelArts-style launcher. Each value may be
# overridden explicitly for another platform.
NNODES=${MA_NUM_HOSTS:-4}
NPUS_PER_NODE=${MA_NUM_GPUS:-8}
NODE_RANK=${VC_TASK_INDEX:-0}
MASTER_RANK=${MASTER_RANK:-0}
MASTER_PORT=${MASTER_PORT:-18920}

MASTER_ADDR_DEFAULT="${MA_VJ_NAME:-localhost}-${MA_TASK_NAME:-job}-${MASTER_RANK}.${MA_VJ_NAME:-local}"
CURRENT_ADDR_DEFAULT="${MA_VJ_NAME:-localhost}-${MA_TASK_NAME:-job}-${NODE_RANK}.${MA_VJ_NAME:-local}"
MASTER_ADDR=${MASTER_ADDR:-$MASTER_ADDR_DEFAULT}
CURRENT_ADDR=${CURRENT_ADDR:-$CURRENT_ADDR_DEFAULT}
WORLD_SIZE=$((NNODES * NPUS_PER_NODE))

export WORLD_SIZE
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=4
export ASCEND_GLOBAL_LOG_LEVEL=2
export TORCH_DISTRIBUTED_DEBUG=INFO
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-64000}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "**** NNODES: ${NNODES}"
echo "**** NPUS_PER_NODE: ${NPUS_PER_NODE}"
echo "**** WORLD_SIZE: ${WORLD_SIZE}"
echo "**** NODE_RANK: ${NODE_RANK}"
echo "**** MASTER_ADDR: ${MASTER_ADDR}"
echo "**** MASTER_PORT: ${MASTER_PORT}"
echo "**** CURRENT_ADDR: ${CURRENT_ADDR}"

# group_idx must already exist in the shared dataset directory. Never generate
# it concurrently from the training nodes.
LOG_FILE="tmp_pretrain_${NAME}_node${NODE_RANK}.txt"

torchrun \
    --nproc_per_node="$NPUS_PER_NODE" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    meta_train_parallel.py \
    --config-name "$CONFIG_NAME" \
    name="$NAME" \
    mode=pretrain \
    data.source=$SOURCE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.eval_batch_size=$TEST_BATCH_SIZE \
    run.gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    run.use_gradient_checkpoint=$USE_GRADIENT_CHECKPOINT \
    resume_global_step=$RESUME_GLOBAL_STEP \
    optim.learning_rate=$LEARNING_RATE \
    metanetwork.type=$TYPE \
    data.conversation_max_length=$CONVERSATION_MAX_LEN \
    data.context_max_length=$CONTEXT_MAX_LEN \
    metanetwork.transformer_cfg.num_layers=$NUM_LAYERS \
    optim.warmup_steps=$WARMUP_STEPS \
    metanetwork.method=$METHOD \
    model.lora_r=$LORA_R \
    model.metalora_r="$METALORA_R" \
    2>&1 | tee "$LOG_FILE"

echo "**** END (node ${NODE_RANK}) ****"

