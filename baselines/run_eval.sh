#!/bin/bash
#SBATCH --account=kumarv
#SBATCH --job-name=ens_eval
#SBATCH --output=logs/ens_eval_%A_%a.txt
#SBATCH --error=logs/ens_eval_err_%A_%a.txt
#SBATCH --time=01:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-1

source ~/.bashrc
conda activate ct-lstm

MODELS=("tam-rl") # "lstm" "ctlstm" "gru" "ctgru" "transformer"
SPLITS=("Koppen" "IGBP") 

NUM_MODELS=${#MODELS[@]}
NUM_SPLITS=${#SPLITS[@]}

SPLIT_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SPLITS))
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SPLITS))

MODEL=${MODELS[$MODEL_IDX]}
SPLIT=${SPLITS[$SPLIT_IDX]}

BATCH_SIZE=128

echo "Job ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Split: $SPLIT"

python evaluate_ensemble.py \
    --model $MODEL \
    --split_type $SPLIT \
    --device cuda