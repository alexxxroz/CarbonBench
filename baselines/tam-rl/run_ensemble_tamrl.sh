#!/bin/bash -l
#SBATCH --account=kumarv
#SBATCH --job-name=tamrl_ens
#SBATCH --output=logs/tamrl_ens_%A_%a.txt
#SBATCH --error=logs/tamrl_ens_err_%A_%a.txt
#SBATCH --time=24:00:00
#SBATCH --partition=kgml03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-19

source ~/.bashrc
conda activate cover_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

SPLITS=("IGBP" "Koppen")
SEEDS=(27 28 29 30 31 32 33 34 35 36)

NUM_SPLITS=${#SPLITS[@]}
NUM_SEEDS=${#SEEDS[@]}

SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
SPLIT_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))

SPLIT=${SPLITS[$SPLIT_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Job ID: $SLURM_ARRAY_TASK_ID"
echo "Model: tam-rl"
echo "Split: $SPLIT"
echo "Seed: $SEED"

cd ~/Desktop/CarbonBench/baselines

python train_single.py \
    --model tam-rl \
    --split_type $SPLIT \
    --seed $SEED \
    --config ./configs/tam-rl_${SPLIT}.yaml \
    --output_dir ./outputs \
    --num_epochs 100 \
    --batch_size 128 \
    --device cuda