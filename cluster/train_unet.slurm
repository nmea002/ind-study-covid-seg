#!/bin/bash
# Train the U-Net baseline on one prepared dataset (same run folder and data as LViT, see prepare_run.sh),
# then predict val + test with the best model and report (train_unet.py -> run_report.py).
# Submit from the cluster/ folder of the project (needs ../train_unet.py, ../evaluate.py, ../run_report.py, ../splits/):
#   sbatch -J unet-qata   train_unet.slurm qata
#   sbatch -J unet-mosmed train_unet.slurm mosmed
# Max 100 epochs, early stop after 20 with no val improvement.
# Output: <project>/runs/<name>/LViT/UNet_runs/<task>/Test_session_<date>/
#   best_model-UNet.pth.tar, log, history.csv, preds_val/, preds_test/, run_config.json, metrics.json,
#   per_image_*.csv, curves.png, examples/
#SBATCH --job-name=unet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/cv_data/users/%u/logs/%x-%j.out
#SBATCH --error=/mnt/cv_data/users/%u/logs/%x-%j.err

set -euo pipefail
NAME="${1:?usage: sbatch train_unet.slurm <qata|mosmed>}"
PROJECT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"
RUN="$PROJECT/runs/$NAME/LViT"

module load cuda/13.0
module load miniconda3/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/lvit"
export MPLBACKEND=Agg

nvidia-smi
TASK=$(grep "^task_name" "$RUN/Config.py" | sed "s/.*= *'\(.*\)'.*/\1/")
echo "run: $RUN | task: $TASK"

python -u "$PROJECT/train_unet.py" --task "$TASK" --lvit_dir "$RUN" --workers 8
