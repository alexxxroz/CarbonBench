#!/bin/bash
#SBATCH --account=kumarv
#SBATCH --job-name=ensemble
#SBATCH --output=logs/ensemble_%A_%a.txt
#SBATCH --error=logs/ensemble_err_%A_%a.txt
#SBATCH --time=03:00:00
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0 #-39

source ~/.bashrc
conda activate ct-lstm

MODELS=("ctlstm") #  "transformer"
SPLITS=("Koppen") #"IGBP" 

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

python train_single_few.py \
    --model $MODEL \
    --split_type $SPLIT \
    --config ./configs/${MODEL}_${SPLIT}.yaml \
    --batch_size $BATCH_SIZE \
    --device cuda