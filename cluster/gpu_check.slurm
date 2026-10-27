#!/bin/bash
# Short test job: confirms the env sees the GPU, a conv runs on it, and BERT loads offline.
# Run before the long training job:   sbatch gpu_check.slurm
#SBATCH --job-name=lvit-gpucheck
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/cv_data/users/%u/logs/%x-%j.out
#SBATCH --error=/mnt/cv_data/users/%u/logs/%x-%j.err

set -euo pipefail
module load cuda/13.0
module load miniconda3/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/lvit"
export HF_HUB_OFFLINE=1

nvidia-smi
python -c "
import torch
print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0), '| capability', torch.cuda.get_device_capability(0))
c = torch.nn.Conv2d(3, 8, 3).cuda(); c(torch.randn(1, 3, 32, 32).cuda()); torch.cuda.synchronize()
print('conv OK | cuDNN', torch.backends.cudnn.version())
"
cd "$SLURM_SUBMIT_DIR"
python -c "
from bert_embedding import BertEmbedding
toks, embs = BertEmbedding()(['Bilateral pulmonary infection, two infected areas, lower left lung.'])[0]
print('BERT OK:', len(toks), 'tokens, dim', embs[0].shape, toks)
"
