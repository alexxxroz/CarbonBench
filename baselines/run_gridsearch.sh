#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=CB_grid
#SBATCH --output=logs/gridsearch_%j.txt
#SBATCH --error=logs/gridsearch_err_%j.txt
#SBATCH --time=24:00:00
#SBATCH --partition=msigpu #kgml03 #kgml02
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-13 # 7 (models) * 2 (splits) = 14 

source ~/.bashrc
conda activate ct-lstm

# Parameters
ARCHITECTURES=("lstm" "ctlstm" "gru" "ctgru" "transformer" "patch_transformer" "tam-rl")
SPLITS=("Koppen" "IGBP")
DEVICE="cuda"

NUM_ARCH=${#ARCHITECTURES[@]}   # 7
NUM_SPLITS=${#SPLITS[@]}         # 2
IDX=${SLURM_ARRAY_TASK_ID}     # 0..13

arch_idx=$(( IDX % NUM_ARCH ))
split_idx=$(( IDX / NUM_ARCH ))

ARCH_VAL=${ARCHITECTURES[$arch_idx]}
SPLIT_VAL=${SPLITS[$split_idx]}


python3 train_gridsearch.py --model "$ARCH_VAL" --split_type "$SPLIT_VAL" --device "$DEVICE"
