#!/bin/bash

#SBATCH --account=bcjw-dtai-gh
#SBATCH --job-name=single_gpu
#SBATCH --partition=ghx4
#SBATCH --mem=732g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=01:00:00
#SBATCH --output=four_chip.log
#SBATCH --error=four_chip.err


MODE="${1:-}"
MODEL_SIZE="${2:-}"
SEQUENCE_LENGTH="${3:-}"
MICRO_BATCH_SIZE="${4:-1}"

VALID_MODES=("prism-ulysses" "ulysses")
if [[ ! " ${VALID_MODES[*]} " =~ " ${MODE} " ]]; then
    echo "Invalid mode: '${MODE}'. Valid modes are: ${VALID_MODES[*]}"
    exit 1
fi

MODEL_SIZE_JSON="${SUPEROFFLOAD_DIR}/experiment/model.json"
if ! [[ -f "$MODEL_SIZE_JSON" ]]; then
    echo "Model size JSON not found: $MODEL_SIZE_JSON"
    exit 1
fi

HIDDEN=$(jq -r --arg model_size "$MODEL_SIZE" '.[$model_size].hidden' "$MODEL_SIZE_JSON")
LAYERS=$(jq -r --arg model_size "$MODEL_SIZE" '.[$model_size].layers' "$MODEL_SIZE_JSON")

echo "########################################################"
echo "MODE: $MODE"
echo "MODEL_SIZE: $MODEL_SIZE"
echo "########################################################"

offload_config=""
if [[ "$MODE" == "prism-ulysses" ]]; then
    offload_config="--cpu-optimizer"
fi


echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
# Setup variables for torchrun rdzv_endpoint
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"

DATA_PATH="${SUPEROFFLOAD_DIR}/training_datasets"

DEEPSPEED_CONFIG="${SUPEROFFLOAD_DIR}/experiment/sequence_length/${MODE}.json"
if ! [[ -f "$DEEPSPEED_CONFIG" ]]; then
    echo "DeepSpeed config not found: $DEEPSPEED_CONFIG"
    exit 1
fi

source ~/.bashrc
conda activate ae
module reset
export NCCL_SOCKET_IFNAME=hsn
module load nccl
module load cuda/12.6.1
export NCCL_DEBUG=WARN
export LOGLEVEL=INFO

srun ~/miniconda3/envs/ae/bin/torchrun --nnodes ${SLURM_NNODES} \
    --nproc_per_node 1 \
    --rdzv_id $RANDOM --rdzv_backend c10d \
    --rdzv_endpoint="$head_node_ip:29500" \
    "${MEGATRON_DIR}/pretrain_gpt.py" \
    --use-flash-attn-v2 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --ds-sequence-parallel-size 4 \
    --num-layers "$LAYERS" \
    --hidden-size "$HIDDEN" \
    --num-attention-heads 32 \
    --seq-length "$SEQUENCE_LENGTH" \
    --loss-scale 12 \
    --max-position-embeddings "$SEQUENCE_LENGTH" \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size 16 \
    --train-iters 100 \
    --lr 5.0e-5 \
    --min-lr 1.e-5 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 400 \
    --eval-interval 200 \
    --data-path "$DATA_PATH/gpt2_text_document" \
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
    --exit-interval 10 \
    --make-vocab-size-divisible-by 256 \
    --deepspeed \
    --deepspeed_config="$DEEPSPEED_CONFIG" \
    --zero-stage=3 \
    --no-pipeline-parallel \
    --checkpoint-activations \
    --deepspeed-activation-checkpointing \
     $offload_config \