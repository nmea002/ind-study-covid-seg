#!/bin/bash
# Train LViT on one prepared dataset, then test the best model.
#   sbatch -J lvit-qata   train_lvit.slurm qata
#   sbatch -J lvit-mosmed train_lvit.slurm mosmed
# Max 100 epochs, early stop after 20 with no val improvement (Config.py).
# Output: ~/projects/lvit/runs/<name>/LViT/<task>/LViT/Test_session_<date>/  (models/, log, test_results.txt)
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
RUN="$HOME/projects/lvit/runs/$NAME/LViT"
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

# 2. Test the newest session's best model (README step 3: test_session goes in Config.py)
SESSION=$(ls "$TASK/LViT" | sort | tail -1)
echo "testing session: $SESSION"
sed -i '/test_session = "Test_session_[0-9.]*_[0-9h]*"$/d' Config.py
# leading \n: Config.py doesn't end with a newline, so a plain append would land inside its last comment
printf '\ntest_session = "%s"\n' "$SESSION" >> Config.py
python -u test_model.py 2>&1 | tee "$TASK/LViT/$SESSION/test_results.txt"
