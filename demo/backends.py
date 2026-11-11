"""Inference-only wrappers for the demo: one validated checkpoint per modality, loaded from demo_config.yaml.

Both backends return a lesion probability map (H x W, floats in 0-1) at the uploaded image's own resolution, using the
same preprocessing and prediction rule as the evaluated runs:

- MedCLIPSeg (medclipseg_setup/predict.py): image read with cv2 (3-channel, BGR as in training), resized to 224x224,
  CLIP-normalized; mean sigmoid probability over 30 stochastic forward passes; resized back bilinearly.
- MedSAM3 (medsam3_setup/common.py): the project's own prob_map(): RGB, 1008x1008, one text/concept query,
  detections with score > 0.5, NMS at IoU 0.5, max over the kept instance masks.

The threshold applied afterwards is the one chosen on val for that checkpoint (demo_config.yaml).

Each upstream repo is put on sys.path (and made the working directory, since both load files by relative path) only
while its model is imported and built. MedCLIPSeg ships a top-level `datasets/` package, which would otherwise shadow
the Hugging Face `datasets` library for the rest of the process.
"""
import contextlib
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42  # same seed for every request, so the same image + prompt always gives the same mask


def project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT, path)


@contextlib.contextmanager
def repo_context(repo_dir, *extra_paths):
    """Temporarily import from (and run inside) an upstream repo folder."""
    old_cwd, old_path = os.getcwd(), list(sys.path)
    sys.path[:0] = [repo_dir, *extra_paths]
    os.chdir(repo_dir)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class MedCLIPSegBackend:
    name = "MedCLIPSeg"

    def __init__(self, spec, device=None):
        self.device = device or default_device()
        repo = project_path(spec.get("repo", "MedCLIPSeg"))
        self.checkpoint = project_path(spec["checkpoint"])
        if not os.path.isfile(self.checkpoint):
            raise FileNotFoundError(f"MedCLIPSeg checkpoint not found: {spec['checkpoint']}")
        with repo_context(repo):
            from trainers import build_medclipseg_unimedclip
            from utils.main_utils import load_cfg_from_cfg_file

            cfg = load_cfg_from_cfg_file(project_path(spec["config"]))
            cfg.MODEL.DEVICE = self.device
            model = build_medclipseg_unimedclip(cfg)  # downloads UniMedCLIP / BiomedBERT on first use
            ckpt = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt["model"])
        self.model = model.eval().to(self.device)
        self.size = cfg.DATASET.SIZE
        self.num_samples = cfg.TEST.NUM_SAMPLES
        self.epoch = ckpt.get("epoch", -1) + 1
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # CLIP_NORMALIZE in their dataloader
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    @torch.no_grad()
    def predict(self, image_path, prompt):
        image = cv2.imread(image_path)  # 3-channel BGR uint8, exactly as their DatasetSegmentation reads it
        if image is None:
            raise ValueError("the file could not be decoded as an image")
        h, w = image.shape[:2]
        x = cv2.resize(image, (self.size, self.size))
        x = torch.from_numpy(x).permute(2, 0, 1).float().div(255)
        x = ((x - self.mean) / self.std).unsqueeze(0).to(self.device)
        torch.manual_seed(SEED)
        samples = self.model(image=x, text=[prompt], num_samples=self.num_samples)  # [N, 1, H, W] logits
        prob = torch.sigmoid(samples).mean(dim=0)
        if prob.ndim == 4:
            prob = prob[:, 0]
        prob = F.interpolate(prob.unsqueeze(1).float(), size=(h, w), mode="bilinear", align_corners=False)
        return prob[0, 0].cpu().numpy()


class MedSAM3Backend:
    name = "MedSAM3"

    def __init__(self, spec, device=None):
        self.device = device or default_device()
        if self.device != "cuda":
            raise RuntimeError("MedSAM3 (SAM3) needs a CUDA GPU; on Colab choose Runtime > Change runtime type > GPU")
        repo = project_path(spec.get("repo", "MedSAM3"))
        self.checkpoint = project_path(spec["checkpoint"])
        if not os.path.isfile(self.checkpoint):
            raise FileNotFoundError(f"MedSAM3 LoRA weights not found: {spec['checkpoint']}")
        with repo_context(repo, os.path.join(PROJECT, "medsam3_setup")):
            import common  # medsam3_setup/common.py: build + the instance rule used in every evaluated run

            self.model = common.build_inference_model(project_path(spec["config"]), self.checkpoint, self.device)
            self._prob_map = common.prob_map
        self.epoch = None

    def predict(self, image_path, prompt):
        from PIL import Image

        with Image.open(image_path) as im:
            image = im.convert("RGB")
        torch.manual_seed(SEED)
        return self._prob_map(self.model, image, prompt, self.device)


BACKENDS = {"medclipseg": MedCLIPSegBackend, "medsam3": MedSAM3Backend}


def load_backend(spec, device=None):
    return BACKENDS[spec["backend"]](spec, device)


def binarize(prob, threshold):
    return (np.asarray(prob) > threshold).astype(np.uint8)
