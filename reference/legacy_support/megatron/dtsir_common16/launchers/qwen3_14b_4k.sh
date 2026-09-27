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

TP=4
PP=2
CP=1
MBS=1
GBS=128
SEQ_LENGTH=4096
TRAIN_ITERS=10
CP_TYPE='ulysses_cp_algo'

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-rotary-position-embeddings \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1.25e-6 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} 
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --sequence-parallel
"

GPT_ARGS="
    --use-mcore-models \
    --qk-layernorm \
    --kv-channels 128 \
    --tokenizer-model ${TOKENIZER_PATH} \
    --max-position-embeddings 40960 \
    --num-layers 40 \
    --hidden-size 5120 \
    --untie-embeddings-and-output-weights \
    --ffn-hidden-size 17408 \
    --num-attention-heads 40 \
    --tokenizer-type HuggingFaceTokenizer \
    --make-vocab-size-divisible-by 1 \
    --rotary-base 1000000 \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --attention-softmax-in-fp32 \

    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 8
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng
"

python -m torch.distributed.run $DISTRIBUTED_ARGS dtsir_common16/entry.py \
    $GPT_ARGS \
    $DATA_ARGS \
     \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    --distributed-backend nccl \
    | tee "${DTSIR_IR_OUT}/node${NODE_RANK}_launcher.log"
