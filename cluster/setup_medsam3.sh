#!/bin/bash
# One-time setup for MedSAM3 on the cluster. Run on the LOGIN node (needs internet), from cluster/:
#   hf auth login        # once, with a Hugging Face token of an account that has access to facebook/sam3 (gated)
#   bash setup_medsam3.sh
# 1. clones MedSAM3 into the project folder (next to cluster/)
# 2. builds the conda env ~/envs/medsam3 (PyTorch for CUDA 13, ARM64 + Blackwell)
# 3. downloads SAM3 (facebook/sam3) and the released MedSAM3 v1 LoRA (lal-Joey/MedSAM3_v1 -> MedSAM3/medsam3_v1_lora.pt)
# 4. unzips our COCO export (model_data/medsam3/*.zip from build_model_data.py)
# 5. (on a GPU node, see the end) build SAM3 + LoRA once and check that every LoRA tensor loads
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(dirname "$HERE")"
ENV="$HOME/envs/medsam3"

module load cuda/13.0
module load miniconda3/latest
source "$(conda info --base)/etc/profile.d/conda.sh"

cd "$PROJECT"
[ -d MedSAM3 ] || git clone https://github.com/Joey-S-Liu/MedSAM3

mkdir -p "$HOME/envs"
[ -d "$ENV" ] || conda create -y -p "$ENV" python=3.12
conda activate "$ENV"
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# their requirements without torch (installed above), decord (video only, no ARM64 build) and dev tools
grep -v -E '^(torch|torchvision|decord|pytest|black|flake8)' MedSAM3/requirements.txt | grep -v '^\s*$' \
    > /tmp/medsam3_requirements_$$.txt
python -m pip install -r /tmp/medsam3_requirements_$$.txt pandas openpyxl torchmetrics  # torchmetrics: sam3 losses
rm /tmp/medsam3_requirements_$$.txt
python -m pip install --no-deps -e MedSAM3
# sam3 imports decord at module level (sam3/train/data/sam3_image_dataset.py) although only video loading uses it:
# install a stub that imports fine and fails loudly if a video is ever read
python - <<'EOF'
import os, site
d = os.path.join(site.getsitepackages()[0], "decord")
os.makedirs(d, exist_ok=True)
with open(os.path.join(d, "__init__.py"), "w") as f:
    f.write('"""Stub: decord has no ARM64 build; SAM3 imports it but image training never reads video."""\n'
            "def _unavailable(*args, **kwargs):\n"
            "    raise ImportError('decord stub: video reading is not available in this env')\n"
            "cpu = gpu = _unavailable\n"
            "class VideoReader:\n"
            "    def __init__(self, *args, **kwargs):\n"
            "        _unavailable()\n")
print("decord stub ->", d)
EOF

cd "$PROJECT/MedSAM3"
python - <<'EOF'
import glob, os, shutil
import torch
from huggingface_hub import snapshot_download
from sam3.model_builder import download_ckpt_from_hf

download_ckpt_from_hf()   # facebook/sam3 (gated): fails here if the account has no access
print("SAM3 checkpoint cached")

folder = snapshot_download("lal-Joey/MedSAM3_v1")
files = [f for ext in ("pt", "pth", "bin", "safetensors") for f in glob.glob(os.path.join(folder, "**", "*." + ext), recursive=True)]
print("MedSAM3_v1 files:", files)
assert files, "no weight file found in lal-Joey/MedSAM3_v1"
src = sorted(files, key=lambda f: ("best" not in os.path.basename(f), f))[0]
if src.endswith(".safetensors"):
    from safetensors.torch import load_file
    torch.save(load_file(src), "medsam3_v1_lora.pt")
else:
    shutil.copy(src, "medsam3_v1_lora.pt")
print("MedSAM3 v1 LoRA ->", os.path.abspath("medsam3_v1_lora.pt"), "from", src)
EOF

cd "$PROJECT/model_data/medsam3"
for z in qata mosmed; do
    [ -d "$z" ] || python -m zipfile -e "$z.zip" .
    echo "$z: $(ls "$z/train" | wc -l) train / $(ls "$z/valid" | wc -l) valid / $(ls "$z/test" | wc -l) test files"
done

# The SAM3 + LoRA build check needs a GPU (sam3's position encoding allocates on "cuda"), so it is not run here.
# On a GPU node, from MedSAM3/:
#   python -c "import sys; sys.path.insert(0, '../medsam3_setup'); from common import build_inference_model; \
#       build_inference_model('../medsam3_setup/configs/qata.yaml', 'medsam3_v1_lora.pt', device='cuda'); print('OK')"

python -c "import torch; print('torch', torch.__version__, '| built for CUDA', torch.version.cuda)"
echo "Done."
