#!/bin/bash

#SBATCH --account=bcjw-dtai-gh
#SBATCH --job-name=rollback
#SBATCH --partition=ghx4
#SBATCH --mem=120g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --output=rollback.log
#SBATCH --error=rollback.err

module reset
source ~/.bashrc
conda activate ae
module load cuda/12.6.1

python3 ${DEEPSPEED_DIR}/tests/superadam/rollback.py
