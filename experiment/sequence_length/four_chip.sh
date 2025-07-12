#!/bin/bash

MODE=$1
MODEL_SIZE=$2
SEQUENCE_LENGTH=$3
ACTIVATION_CHECKPOINTING=$4

if [[ "$MODE" != "prism-ulysses" && "$MODE" != "ulysses" ]]; then
    echo "Invalid mode. Please use 'prism-ulysses' or 'ulysses'."
    exit 1
fi

MODEL_SIZE_JSON="../model.json"
HIDDEN=$(jq -r --arg model_size "$MODEL_SIZE" '.[$model_size].hidden' $MODEL_SIZE_JSON)
LAYERS=$(jq -r --arg model_size "$MODEL_SIZE" '.[$model_size].layers' $MODEL_SIZE_JSON)

DATA_PATH="../../training_datasets"

echo "MODEL_SIZE: $MODEL_SIZE"
echo "SEQUENCE_LENGTH: $SEQUENCE_LENGTH"
echo "ACTIVATION_CHECKPOINTING: $ACTIVATION_CHECKPOINTING"  
echo "HIDDEN: $HIDDEN"
echo "LAYERS: $LAYERS"

if ${ACTIVATION_CHECKPOINTING} == "true"; then
    ac_config="--checkpoint-activations --deepspeed-activation-checkpointing"
else
    ac_config=""
fi

if ${MODE} == "prism-ulysses"; then
    offload_config="--cpu-optimizer"
else
    offload_config=""
fi

deepspeed --num_nodes 2 --num_gpus 2 --bind_cores_to_rank ${Megatron-DeepSpeed}/pretrain_gpt.py --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --ds-sequence-parallel-size 4 --num-layers $LAYERS --hidden-size $HIDDEN --num-attention-heads 32 --seq-length $SEQUENCE_LENGTH --loss-scale 12 --max-position-embeddings $SEQUENCE_LENGTH --micro-batch-size 1 --global-batch-size 1 --train-iters 100 --lr 5.0e-5 --min-lr 1.e-5 --lr-decay-style cosine --log-interval 1 --eval-iters 400 --eval-interval 200 --data-path $DATA_PATH/my-gpt2_text_document --vocab-file $DATA_PATH/gpt2-vocab.json --merge-file $DATA_PATH/gpt2-merges.txt --save-interval 100 --split 98,2,0 --clip-grad 1.0 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.006 --fp16 --exit-interval 100 --make-vocab-size-divisible-by 256 $offload_config --deepspeed --deepspeed_config=${MODE}.json --zero-stage=3 --no-pipeline-parallel $ac_config
