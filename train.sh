#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/train_%A.out
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="vram40|vram48|vram80"
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4

# --- Conda ---
module load conda/latest
conda activate ragenv

module load cuda/12.6

nvidia-smi

python train.py
