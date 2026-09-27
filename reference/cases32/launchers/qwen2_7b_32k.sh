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
TP=2
PP=2
CP=2
CP_ALGO=ulysses_cp_algo

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
    --hidden-size 3584 \
    --ffn-hidden-size 18944 \
    --num-attention-heads 28 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_PATH} \
    --seq-length 32768 \
    --max-position-embeddings 32768 \
    --micro-batch-size 2 \
    --global-batch-size 64 \
    --make-vocab-size-divisible-by 1 \
    --rotary-base 1000000 \
    --lr 1.25e-6 \
    --train-iters 10 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
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
    --bf16 \
    --group-query-attention \
    --num-query-groups 4 \
    --rope-scaling-factor 8 \
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
