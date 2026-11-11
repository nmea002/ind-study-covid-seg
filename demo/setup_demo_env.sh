#!/bin/bash
# Installs everything the demo needs into the CURRENT Python environment (a Colab GPU runtime, or an activated
# conda env on the cluster), from the project root:
#   hf auth login            # once; account must have access to the gated facebook/sam3 model
#   bash demo/setup_demo_env.sh
#
# 1. clones MedCLIPSeg and MedSAM3 next to demo/ (unchanged upstream code, as for training)
# 2. installs both models' inference dependencies into one environment (torch already present is kept)
# 3. installs a decord stub (SAM3 imports decord at module level; images never use it)
# 4. downloads SAM3 and the released MedSAM3 v1 LoRA -> MedSAM3/medsam3_v1_lora.pt
#    (UniMedCLIP and BiomedBERT for MedCLIPSeg are downloaded automatically the first time it is loaded)
# The fine-tuned MedCLIPSeg checkpoint is NOT downloaded here: copy it to demo/weights/ (see DEMO_README.md).
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d MedCLIPSeg ] || git clone --depth 1 https://github.com/HealthX-Lab/MedCLIPSeg
[ -d MedSAM3 ] || git clone --depth 1 https://github.com/Joey-S-Liu/MedSAM3

python -c "import torch" 2>/dev/null || python -m pip install torch torchvision
# MedCLIPSeg model code: transformers pinned as in their requirements.txt (also satisfies MedSAM3's >=4.48)
python -m pip install -q "transformers==4.56.0" timm open_clip_torch ftfy regex sentencepiece einops \
    huggingface_hub pandas openpyxl pyyaml tqdm opencv-python-headless scipy
# MedSAM3 / SAM3: their requirements.txt without torch, decord (video only) and dev tools
python -m pip install -q pycocotools hydra-core omegaconf submitit iopath torchmetrics psutil scikit-image \
    scikit-learn matplotlib "gradio>=5,<6"  # gradio 6 needs huggingface_hub>=1.16, transformers 4.56 needs <1.0
python -m pip install -q --no-deps -e MedSAM3

python - <<'EOF'
import importlib.util, os, site
if importlib.util.find_spec("decord") is None:
    d = os.path.join(site.getsitepackages()[0], "decord")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "__init__.py"), "w") as f:
        f.write('"""Stub: SAM3 imports decord, but image inference never reads video."""\n'
                "def _unavailable(*args, **kwargs):\n"
                "    raise ImportError('decord stub: video reading is not available in this env')\n"
                "cpu = gpu = _unavailable\n"
                "class VideoReader:\n"
                "    def __init__(self, *args, **kwargs):\n"
                "        _unavailable()\n")
    print("decord stub ->", d)
EOF

cd MedSAM3
python - <<'EOF'
import glob, os, shutil
import torch
from huggingface_hub import snapshot_download
from sam3.model_builder import download_ckpt_from_hf

download_ckpt_from_hf()  # facebook/sam3 (gated): fails here without `hf auth login` + access
print("SAM3 checkpoint cached")
if not os.path.exists("medsam3_v1_lora.pt"):
    folder = snapshot_download("lal-Joey/MedSAM3_v1")
    files = [f for ext in ("pt", "pth", "bin", "safetensors")
             for f in glob.glob(os.path.join(folder, "**", "*." + ext), recursive=True)]
    assert files, "no weight file found in lal-Joey/MedSAM3_v1"
    src = sorted(files, key=lambda f: ("best" not in os.path.basename(f), f))[0]  # same pick as setup_medsam3.sh
    if src.endswith(".safetensors"):
        from safetensors.torch import load_file
        torch.save(load_file(src), "medsam3_v1_lora.pt")
    else:
        shutil.copy(src, "medsam3_v1_lora.pt")
print("MedSAM3 v1 LoRA:", os.path.abspath("medsam3_v1_lora.pt"))
EOF
cd ..
mkdir -p demo/weights
ls demo/weights
echo "Done. Put the MedCLIPSeg checkpoint in demo/weights/, then: python demo/app.py --share"
