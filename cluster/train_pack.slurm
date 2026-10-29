#!/bin/bash
# Run several trainings at once in ONE job on one GPU (the cluster allows only 2 gpu jobs per user, and LViT /
# U-Net at batch size 2 leave the GPU mostly idle). Each run is one of the train_<model>.slurm scripts started with
# bash, with its own log. Submit from the cluster/ folder:
#   sbatch -J baselines train_pack.slurm lvit:qata lvit:mosmed unet:qata unet:mosmed
#   sbatch -J mcs       train_pack.slurm medclipseg:qata:sanity medclipseg:qata medclipseg:mosmed
# Logs: /mnt/cv_data/users/<you>/logs/pack-<jobid>-<model>-<dataset>[-<extra>].log
#SBATCH --job-name=pack
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00   # max on the gpu partition
#SBATCH --output=/mnt/cv_data/users/%u/logs/%x-%j.out
#SBATCH --error=/mnt/cv_data/users/%u/logs/%x-%j.err

set -uo pipefail   # no -e: one failing run must not stop the others
[ $# -gt 0 ] || { echo "usage: sbatch train_pack.slurm <model>:<dataset>[:<extra>] ..."; exit 1; }
LOGS="/mnt/cv_data/users/$USER/logs"
cd "$SLURM_SUBMIT_DIR"
nvidia-smi

declare -A PIDS
for spec in "$@"; do
    IFS=: read -r model dataset extra <<< "$spec"
    log="$LOGS/pack-$SLURM_JOB_ID-${spec//:/-}.log"
    echo "starting $spec -> $log"
    bash "train_$model.slurm" "$dataset" ${extra:+"$extra"} > "$log" 2>&1 &
    PIDS[$spec]=$!
    sleep 60   # stagger the starts (data loading, BERT embedding, weight loading)
done

status=0
for spec in "${!PIDS[@]}"; do
    if wait "${PIDS[$spec]}"; then echo "finished: $spec"; else echo "FAILED: $spec (see its log)"; status=1; fi
done
exit $status
