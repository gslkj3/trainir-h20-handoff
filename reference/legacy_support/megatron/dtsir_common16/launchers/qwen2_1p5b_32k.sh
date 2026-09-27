#!/bin/bash
set -eo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6002}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-0}

DATA_PATH="${DTSIR_DATA_PATH:?Missing dataset prefix}"
TOKENIZER_PATH="${DTSIR_TOKENIZER_PATH:?Missing tokenizer}"

TP=1
PP=1
CP=4
CP_ALGO=megatron_cp_algo
MBS=1
GBS=64
SEQ_LEN=32768

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --sequence-parallel \
    --num-layers 28 \
    --hidden-size 1536 \
    --ffn-hidden-size 8960 \
    --num-attention-heads 12 \
    --group-query-attention \
    --num-query-groups 2 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --rotary-base 1000000 \
    --train-iters 10 \
    --lr 1.25e-6 \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --norm-epsilon 1e-06 \
    --swiglu \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-rotary-position-embeddings \
    --overlap-grad-reduce \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --add-qkv-bias \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --rope-scaling-factor 8 \
    --seed 42 \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 10,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10 \
    --eval-interval 10 \
    --eval-iters 0 \
"

python -m torch.distributed.run $DISTRIBUTED_ARGS dtsir_common16/entry.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    | tee "${DTSIR_IR_OUT}/node${NODE_RANK}_launcher.log"
