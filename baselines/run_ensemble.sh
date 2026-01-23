#!/bin/bash
#SBATCH --account=kumarv
#SBATCH --job-name=ensemble
#SBATCH --output=logs/ensemble_%A_%a.txt
#SBATCH --error=logs/ensemble_err_%A_%a.txt
#SBATCH --time=24:00:00
#SBATCH --partition=kgml02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0 #-99

#export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
source ~/.bashrc
#conda activate cover_env
conda activate ct-lstm

MODELS=("lstm") # "ctlstm" "gru" "ctgru" "transformer")
SPLITS=("IGBP") # "Koppen")

# suitable seeds?
SEEDS=(27) # 28 29 30 31 32 33 34 35 36)

NUM_MODELS=${#MODELS[@]}
NUM_SPLITS=${#SPLITS[@]}
NUM_SEEDS=${#SEEDS[@]}

SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
TEMP=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SPLIT_IDX=$((TEMP % NUM_SPLITS))
MODEL_IDX=$((TEMP / NUM_SPLITS))

MODEL=${MODELS[$MODEL_IDX]}
SPLIT=${SPLITS[$SPLIT_IDX]}
SEED=${SEEDS[$SEED_IDX]}

BATCH_SIZE=128
EPOCHS=100
echo "Job ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Split: $SPLIT"
echo "Seed: $SEED"

cd ~/Desktop/CarbonBench/baselines

python train_single.py \
    --model $MODEL \
    --split_type $SPLIT \
    --seed $SEED \
    --config ./configs/${MODEL}_${SPLIT}.yaml \
    --output_dir ./outputs \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --device cuda