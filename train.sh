#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/train_%A.out
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="vram40|vram48|vram80"
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL

# --- Conda ---
module load conda/latest
conda activate ragenv

module load cuda/12.6

nvidia-smi

python train.py \
    --model "google-bert/bert-base-uncased" \
    --batch_size 128 \
    --lr 2e-5 \
    --weight_decay 0.01 \
    --epochs 5 \
    --beta 0 \
    --min_rel 3 \
    --num_workers 4 \
    --ckpt_interval 1000 \
    --output "bi_encoder.pth" \
    --wandb_project "self-dist-retrieval"
