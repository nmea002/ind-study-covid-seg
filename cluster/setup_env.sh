#!/bin/bash
# One-time setup of the Python environment for LViT on the DGX Spark cluster (~10 min).
# Run on the LOGIN node (needs internet, no GPU):   bash setup_env.sh
#
# Not the Colab recipe: the Spark is ARM64 + Blackwell GPU, so Python 3.7 / torch 1.8+cu111 / mxnet
# (bert-embedding) have no builds for it. Instead: Python 3.11, current PyTorch built for CUDA 13,
# and bert_embedding.py (same BERT weights via HuggingFace) in place of the bert-embedding package.
set -euo pipefail

ENV="$HOME/envs/lvit"

module load cuda/13.0
module load miniconda3/latest
source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "$HOME/envs"
[ -d "$ENV" ] || conda create -y -p "$ENV" python=3.11
conda activate "$ENV"

python -m pip install --upgrade pip
# PyTorch for CUDA 13 (has ARM64 + Blackwell builds)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# Only the packages LViT actually imports (requirements.txt also lists many it never uses).
# pandas<3: utils.read_text pads short descriptions via chained assignment, which pandas 3 silently ignores.
# openpyxl: newer pandas needs it to read the .xlsx text files.
python -m pip install \
    "pandas<3" openpyxl numpy scipy scikit-learn opencv-python-headless Pillow matplotlib \
    tensorboardX thop torchinfo timm ml-collections einops tqdm transformers

# Download the BERT weights now: compute nodes may not have internet.
python -c "from transformers import BertModel, BertTokenizer; BertTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased'); print('BERT weights cached')"

python -c "import torch; print('torch', torch.__version__, '| built for CUDA', torch.version.cuda)"
echo "Done. (No GPU on the login node, so CUDA is checked by gpu_check.slurm.)"
