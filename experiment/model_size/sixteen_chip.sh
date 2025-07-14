#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${1:-}"
MODEL_SIZE="${2:-}"

VALID_MODES=("prism" "zero_offload" "zero_infinity")
if [[ ! " ${VALID_MODES[*]} " =~ " ${MODE} " ]]; then
    echo "Invalid mode: '${MODE}'. Valid modes are: ${VALID_MODES[*]}"
    exit 1
fi

MODEL_SIZE_JSON="${SCRIPT_DIR}/../model.json"
if [[ ! -f "$MODEL_SIZE_JSON" ]]; then
    echo "Model size JSON not found: $MODEL_SIZE_JSON"
    exit 1
fi

HIDDEN=$(jq -r --arg model_size "$MODEL_SIZE" '.[$model_size].hidden' "$MODEL_SIZE_JSON")
LAYERS=$(jq -r --arg model_size "$MODEL_SIZE" '.[$model_size].layers' "$MODEL_SIZE_JSON")

if [[ -z "$HIDDEN" || -z "$LAYERS" || "$HIDDEN" == "null" || "$LAYERS" == "null" ]]; then
    echo "Invalid MODEL_SIZE: '$MODEL_SIZE'. Please check $MODEL_SIZE_JSON."
    exit 1
fi

DATA_PATH="${SCRIPT_DIR}/../../training_datasets"

echo "MODEL_SIZE: $MODEL_SIZE"
echo "HIDDEN: $HIDDEN"
echo "LAYERS: $LAYERS"

DEEPSPEED_CONFIG="${SCRIPT_DIR}/${MODE}.json"
if [[ ! -f "$DEEPSPEED_CONFIG" ]]; then
    echo "DeepSpeed config not found: $DEEPSPEED_CONFIG"
    exit 1
fi

deepspeed \
    --num_nodes 8 \
    --num_gpus 2 \
    --bind_cores_to_rank \
    "${MEGATRON_DEEPSPEED_DIR}/pretrain_gpt.py" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --ds-sequence-parallel-size 1 \
    --num-layers "$LAYERS" \
    --hidden-size "$HIDDEN" \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --loss-scale 12 \
    --max-position-embeddings 2048 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --train-iters 100 \
    --lr 5.0e-5 \
    --min-lr 1.e-5 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 400 \
    --eval-interval 200 \
    --data-path "$DATA_PATH/my-gpt2_text_document" \
    --vocab-file "$DATA_PATH/gpt2-vocab.json" \
    --merge-file "$DATA_PATH/gpt2-merges.txt" \
    --save-interval 100 \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --fp16 \
    --exit-interval 100 \
    --make-vocab-size-divisible-by 256 \
    --cpu-optimizer \
    --deepspeed \
    --deepspeed_config="$DEEPSPEED_CONFIG" \
    --zero-stage=3 \
    --no-pipeline-parallel \
    --checkpoint-activations \
    --deepspeed-activation-checkpointing
