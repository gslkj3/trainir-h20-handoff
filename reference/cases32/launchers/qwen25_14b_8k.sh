#!/bin/bash
set -eo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6002}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${NNODES:-4}
NODE_RANK=${NODE_RANK:-0}

DATA_PATH="${DTSIR_DATA_PATH:?}"
TOKENIZER_PATH="${DTSIR_TOKENIZER_PATH:?}"

TP=8
PP=1
CP=1
CP_ALGO=megatron_cp_algo
MBS=2
GBS=64
SEQ_LEN=8192
TRAIN_ITERS=10

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
    --sequence-parallel \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 6 \
    --num-layers 48 \
    --hidden-size 5120 \
    --ffn-hidden-size 13824 \
    --num-attention-heads 40 \
    --group-query-attention \
    --num-query-groups 8 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --rotary-base 1000000 \
    --lr 1.25e-6 \
    --train-iters ${TRAIN_ITERS} \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --swiglu \
    --use-flash-attn \
    --use-rotary-position-embeddings \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --add-qkv-bias \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --seed 42 \
    --reset-position-ids \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
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

python -m torch.distributed.run $DISTRIBUTED_ARGS dtsir_common32/entry.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    | tee "${DTSIR_IR_OUT}/node${NODE_RANK}_launcher.log"
