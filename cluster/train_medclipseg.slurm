#!/bin/bash
# MedCLIPSeg on one dataset (needs setup_medclipseg.sh first). Submit from the cluster/ folder of the project:
#   sbatch -J mcs-sanity train_medclipseg.slurm qata sanity   # released QaTa checkpoint on our val split, no training
#   sbatch -J mcs-qata   train_medclipseg.slurm qata          # train (resumes if a run exists), then predict + report
#   sbatch -J mcs-mosmed train_medclipseg.slurm mosmed
#   BUDGET_HOURS=5 sbatch -J mcs-qata train_medclipseg.slurm qata final   # deadline: train at most 5 h (resuming),
#                                                       # then predict + report with the best checkpoint so far
# Max 50 epochs, early stop after 20 with no val improvement. Training stops itself before the 12 h limit and this
# script resubmits itself to continue (--resume); if the resubmit is refused (2-job limit), submit the same command
# again by hand later. Prediction runs once TRAINING_DONE exists, in a fresh job if little time is left.
# Output: <project>/runs/medclipseg/<Covid19|MosMedPlus>/trained_models/seed42/
#   checkpoints, log.txt, history.csv, preds_val/, preds_test/, run_config.json, metrics.json, per_image_*.csv,
#   curves.png, examples/
#SBATCH --job-name=medclipseg
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00   # max on the gpu partition
#SBATCH --output=/mnt/cv_data/users/%u/logs/%x-%j.out
#SBATCH --error=/mnt/cv_data/users/%u/logs/%x-%j.err

set -euo pipefail
DATASET="${1:?usage: sbatch train_medclipseg.slurm <qata|mosmed> [sanity]}"
case "$DATASET" in
    qata) NAME=Covid19 ;;
    mosmed) NAME=MosMedPlus ;;
    *) echo "unknown dataset $DATASET"; exit 1 ;;
esac
PROJECT="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"
CONFIG="../medclipseg_setup/configs/$NAME.yaml"
RUN_DIR="$PROJECT/runs/medclipseg/$NAME/trained_models/seed42"

module load cuda/13.0
module load miniconda3/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/medclipseg"
export MPLBACKEND=Agg

nvidia-smi
cd "$PROJECT/MedCLIPSeg"

if [ "${2:-}" = "sanity" ]; then
    python -u ../medclipseg_setup/predict.py --config-file "$CONFIG" --sanity --seed 42 \
        --checkpoint outputs_medclipseg/Covid19/trained_models/seed666/MedCLIPSeg_unimedclip_ViT-B-16_latest.pth \
        --out_dir "$PROJECT/runs/medclipseg_sanity/$NAME"
    exit 0
fi

if [ ! -f "$RUN_DIR/TRAINING_DONE" ]; then
    python -u ../medclipseg_setup/train.py --config-file "$CONFIG" --output-dir ../runs/medclipseg --seed 42 --resume \
        --time_budget_hours "${BUDGET_HOURS:-11}"
fi

if [ "${2:-}" = "final" ] || { [ -f "$RUN_DIR/TRAINING_DONE" ] && [ "$SECONDS" -lt $((8 * 3600)) ]; }; then
    python -u ../medclipseg_setup/predict.py --config-file "$CONFIG" --output-dir ../runs/medclipseg --seed 42
else
    # training hit its time budget, or finished too late to also predict: continue in a new job
    cd "$PROJECT/cluster"
    sbatch -J "$SLURM_JOB_NAME" train_medclipseg.slurm "$DATASET" \
        || echo "Resubmit refused (job limit?). Run again by hand: sbatch -J $SLURM_JOB_NAME train_medclipseg.slurm $DATASET"
fi
