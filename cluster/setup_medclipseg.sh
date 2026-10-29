#!/bin/bash
# One-time setup for MedCLIPSeg on the cluster. Run on the LOGIN node (needs internet, no GPU), from cluster/:
#   bash setup_medclipseg.sh
# 1. clones MedCLIPSeg into the project folder (next to cluster/)
# 2. builds the conda env ~/envs/medclipseg (Python 3.10 as in their INSTALL.md, but PyTorch for CUDA 13 since the
#    Spark is ARM64 + Blackwell; their pinned torch 2.9.0+cu126 has no build for it)
# 3. downloads what training needs, since compute nodes may have no internet: the UniMedCLIP weights, the
#    BiomedBERT text encoder, and their released QaTa checkpoint (for the sanity check)
# 4. unzips our exported data (model_data/medclipseg/*.zip from build_model_data.py)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(dirname "$HERE")"
ENV="$HOME/envs/medclipseg"

module load cuda/13.0
module load miniconda3/latest
source "$(conda info --base)/etc/profile.d/conda.sh"

cd "$PROJECT"
[ -d MedCLIPSeg ] || git clone https://github.com/HealthX-Lab/MedCLIPSeg

mkdir -p "$HOME/envs"
[ -d "$ENV" ] || conda create -y -p "$ENV" python=3.10
conda activate "$ENV"
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# their requirements without the torch pins (already installed above)
grep -v -E '^(torch|torchvision)==' MedCLIPSeg/requirements.txt > /tmp/medclipseg_requirements_$$.txt
python -m pip install -r /tmp/medclipseg_requirements_$$.txt
rm /tmp/medclipseg_requirements_$$.txt

cd "$PROJECT/MedCLIPSeg"
python -c "from trainers.medclipseg_unimedclip import download_checkpoint; download_checkpoint('unimed_clip_vit_b16.pt')"
python -c "
from transformers import AutoModel, AutoTokenizer
name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
AutoTokenizer.from_pretrained(name); AutoModel.from_pretrained(name); print('BiomedBERT cached')"
python utils/download_ckpt.py --dataset Covid19 --data_percentage 100 --seed 666

cd "$PROJECT/model_data/medclipseg"
for z in Covid19 MosMedPlus; do
    [ -d "$z" ] || python -m zipfile -e "$z.zip" .
    echo "$z: $(ls "$z/Train_Folder/img" | wc -l) train / $(ls "$z/Val_Folder/img" | wc -l) val / $(ls "$z/Test_Folder/img" | wc -l) test images"
done

python -c "import torch; print('torch', torch.__version__, '| built for CUDA', torch.version.cuda)"
echo "Done. Next: sbatch -J mcs-sanity train_medclipseg.slurm qata sanity"
