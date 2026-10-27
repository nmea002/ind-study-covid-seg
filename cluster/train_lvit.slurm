#!/bin/bash
# Train LViT on one prepared dataset, then predict val + test with the best model and report (predict_lvit.py).
# Submit from the cluster/ folder of the project (predict_lvit.py needs ../evaluate.py, ../run_report.py, ../splits/):
#   sbatch -J lvit-qata   train_lvit.slurm qata
#   sbatch -J lvit-mosmed train_lvit.slurm mosmed
# Max 100 epochs, early stop after 20 with no val improvement (Config.py).
# Output: <project>/runs/<name>/LViT/<task>/LViT/Test_session_<date>/
#   models/, log, history.csv, preds_val/, preds_test/, run_config.json, metrics.json, per_image_*.csv,
#   curves.png, examples/, report.txt
#SBATCH --job-name=lvit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/cv_data/users/%u/logs/%x-%j.out
#SBATCH --error=/mnt/cv_data/users/%u/logs/%x-%j.err

set -euo pipefail
NAME="${1:?usage: sbatch train_lvit.slurm <qata|mosmed>}"
PROJECT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"
RUN="$PROJECT/runs/$NAME/LViT"
cd "$RUN"

module load cuda/13.0
module load miniconda3/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/lvit"
export HF_HUB_OFFLINE=1   # BERT weights were cached by setup_env.sh
export MPLBACKEND=Agg

nvidia-smi
TASK=$(grep "^task_name" Config.py | sed "s/.*= *'\(.*\)'.*/\1/")
echo "run: $RUN | task: $TASK"

# 1. Train (skipped with a 2nd argument "test": sbatch train_lvit.slurm mosmed test)
if [ "${2:-}" != "test" ]; then
    python -u train_model.py
fi

# 2. Predict val + test with the newest session's best model, pick the threshold on val, report (run_report.py)
SESSION=$(ls "$TASK/LViT" | sort | tail -1)
echo "session: $SESSION"
python -u "$PROJECT/cluster/predict_lvit.py" --project_dir "$PROJECT" --session "$SESSION" 2>&1 \
    | tee "$TASK/LViT/$SESSION/report.txt"
