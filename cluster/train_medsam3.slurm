#!/bin/bash
# MedSAM3 on one dataset (needs setup_medsam3.sh first). Submit from the cluster/ folder of the project, or put it in
# a pack (train_pack.slurm medsam3:qata:zeroshot medsam3:qata ...):
#   sbatch -J ms3-zs-qata train_medsam3.slurm qata zeroshot   # released MedSAM3 LoRA, text prompt only, val + test
#   sbatch -J ms3-qata    train_medsam3.slurm qata            # LoRA fine-tuning from it, then predict + report
# Fine-tuning: max 50 epochs, early stop after 20 without a new best val Dice, and a time budget (train.py) so the
# best checkpoint is still predicted and reported inside the 12 h job (BUDGET_HOURS=<h> sbatch ... to set it).
# Output: <project>/runs/medsam3/<dataset>/{zeroshot,finetune}/
#SBATCH --job-name=medsam3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00   # max on the gpu partition
#SBATCH --output=/mnt/cv_data/users/%u/logs/%x-%j.out
#SBATCH --error=/mnt/cv_data/users/%u/logs/%x-%j.err

set -euo pipefail
DATASET="${1:?usage: sbatch train_medsam3.slurm <qata|mosmed> [zeroshot]}"
PROJECT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"

module load cuda/13.0
module load miniconda3/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/medsam3"
export MPLBACKEND=Agg

nvidia-smi
cd "$PROJECT/MedSAM3"
if [ "${2:-}" = "zeroshot" ]; then
    python -u ../medsam3_setup/predict.py --dataset "$DATASET"
else
    python -u ../medsam3_setup/train.py --dataset "$DATASET" --time_budget_hours "${BUDGET_HOURS:-9.5}"
fi
