#!/bin/bash

MODE=$1
MODEL_SIZE=$2
MICRO_BATCH=$3
ACTIVATION_CHECKPOINTING=$4

if [[ "$MODE" != "prism" && "$MODE" != "zero_offload" && "$MODE" != "zero_infinity" ]]; then
    echo "Invalid mode. Please use 'prism', 'zero_offload', or 'zero_infinity'."
    exit 1
fi

MODEL_SIZE_JSON="../model.json"
HIDDEN=$(jq -r --arg model_size "$MODEL_SIZE" '.[$model_size].hidden' $MODEL_SIZE_JSON)
LAYERS=$(jq -r --arg model_size "$MODEL_SIZE" '.[$model_size].layers' $MODEL_SIZE_JSON)

DATA_PATH="../../training_datasets"

echo "MODEL_SIZE: $MODEL_SIZE"
echo "MICRO_BATCH: $MICRO_BATCH"
echo "ACTIVATION_CHECKPOINTING: $ACTIVATION_CHECKPOINTING"  
echo "HIDDEN: $HIDDEN"
echo "LAYERS: $LAYERS"

if ${ACTIVATION_CHECKPOINTING} == "true"; then
    ac_config="--checkpoint-activations --deepspeed-activation-checkpointing"
else
    ac_config=""
fi

deepspeed --num_nodes 1 --num_gpus 1 --bind_cores_to_rank ${Megatron-DeepSpeed}/pretrain_gpt.py --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --ds-sequence-parallel-size 1 --num-layers $LAYERS --hidden-size $HIDDEN --num-attention-heads 32 --seq-length 2048 --loss-scale 12 --max-position-embeddings 2048 --micro-batch-size $MICRO_BATCH --global-batch-size 8 --train-iters 100 --lr 5.0e-5 --min-lr 1.e-5 --lr-decay-style cosine --log-interval 1 --eval-iters 400 --eval-interval 200 --data-path $DATA_PATH/my-gpt2_text_document --vocab-file $DATA_PATH/gpt2-vocab.json --merge-file $DATA_PATH/gpt2-merges.txt --save-interval 100 --split 98,2,0 --clip-grad 1.0 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.006 --fp16 --exit-interval 100 --make-vocab-size-divisible-by 256 --cpu-optimizer --deepspeed --deepspeed_config=${MODE}.json --zero-stage=3 --no-pipeline-parallel $ac_config
