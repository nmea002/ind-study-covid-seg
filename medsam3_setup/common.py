"""Shared MedSAM3 pieces for predict.py and train.py. Imported from inside the MedSAM3/ repo folder (cwd), so SAM3 and
the LoRA code are the repo's own.

Prediction follows MedSAM3/infer_sam.py (SAM3LoRAInference.predict): image resized to 1008x1008, one text/concept
query, detections kept if score > DETECTION_THRESHOLD, then NMS on their boxes at NMS_IOU.
Rule for combining the instances into one lesion map (the same for every image, model and dataset): the probability
of a pixel is the maximum mask probability (sigmoid) over the kept instances, resized to the image size (bilinear);
0 where nothing was detected. run_report.py then picks the pixel threshold on val.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image as PILImage
from torchvision.ops import nms

from lora_layers import LoRAConfig, apply_lora_to_model
from sam3.model.utils.misc import copy_data_to_device
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Datapoint, FindQueryLoaded, InferenceMetadata
from sam3.train.data.sam3_image_dataset import Image as SAMImage
from sam3.train.transforms.basic_for_api import ComposeAPI, NormalizeAPI, RandomResizeAPI, ToTensorAPI

RESOLUTION = 1008
DETECTION_THRESHOLD = 0.5  # infer_sam.py default
NMS_IOU = 0.5              # infer_sam.py default
COCO_FOLDERS = {"train": "train", "val": "valid", "test": "test"}
RELEASED_LORA = "medsam3_v1_lora.pt"  # copied into MedSAM3/ by setup_medsam3.sh


def lora_config(cfg, inference):
    c = cfg["lora"]
    return LoRAConfig(rank=c["rank"], alpha=c["alpha"], dropout=0.0 if inference else c["dropout"],
                      target_modules=c["target_modules"], apply_to_vision_encoder=c["apply_to_vision_encoder"],
                      apply_to_text_encoder=c["apply_to_text_encoder"],
                      apply_to_geometry_encoder=c["apply_to_geometry_encoder"],
                      apply_to_detr_encoder=c["apply_to_detr_encoder"],
                      apply_to_detr_decoder=c["apply_to_detr_decoder"],
                      apply_to_mask_decoder=c["apply_to_mask_decoder"])


def load_lora_checked(model, path):
    """MedSAM3's load_lora_weights uses strict=False, which silently skips names that don't match.
    Here every saved LoRA tensor must land in the model."""
    state = torch.load(path, map_location="cpu")
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model_keys = set(model.state_dict())
    missing = [k for k in state if k not in model_keys]
    assert not missing, f"{len(missing)} of {len(state)} LoRA tensors in {path} don't match the model, e.g. {missing[:3]}"
    model.load_state_dict(state, strict=False)
    print(f"loaded {len(state)} LoRA tensors from {path}", flush=True)


def build_inference_model(config_path, weights_path, device="cuda"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model = build_sam3_image_model(device=device, compile=False, load_from_HF=True,
                                   bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz", eval_mode=True)
    model = apply_lora_to_model(model, lora_config(cfg, inference=True))
    load_lora_checked(model, weights_path)
    return model.to(device).eval()


TRANSFORM = ComposeAPI(transforms=[RandomResizeAPI(sizes=RESOLUTION, max_size=RESOLUTION, square=True,
                                                   consistent_transform=False),
                                   ToTensorAPI(), NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def _datapoint(pil_image, prompt):
    w, h = pil_image.size
    query = FindQueryLoaded(query_text=prompt, image_id=0, object_ids_output=[], is_exhaustive=True,
                            query_processing_order=0,
                            inference_metadata=InferenceMetadata(coco_image_id=0, original_image_id=0,
                                                                 original_category_id=1, original_size=[w, h],
                                                                 object_id=0, frame_index=0))
    return Datapoint(find_queries=[query], images=[SAMImage(data=pil_image, objects=[], size=[h, w])])


@torch.no_grad()
def prob_map(model, pil_image, prompt, device="cuda"):
    """Lesion probability map (H x W, 0-1) of one image for one text prompt (rule in the module docstring)."""
    w, h = pil_image.size
    batch = collate_fn_api([TRANSFORM(_datapoint(pil_image, prompt))], dict_key="input")["input"]
    out = model(copy_data_to_device(batch, torch.device(device), non_blocking=True))[-1]
    scores = out["pred_logits"].sigmoid()[0].max(dim=-1)[0]   # [queries]
    keep = scores > DETECTION_THRESHOLD
    if not keep.any():
        return np.zeros((h, w), dtype=np.float32)
    cx, cy, bw, bh = out["pred_boxes"][0, keep].unbind(-1)
    boxes = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], dim=-1)
    kept = nms(boxes.float(), scores[keep].float(), NMS_IOU)
    masks = out["pred_masks"][0, keep][kept].sigmoid().float()  # [n, 288, 288]
    masks = F.interpolate(masks.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)[0]
    return masks.max(dim=0)[0].cpu().numpy()


def coco_images(data_dir, split):
    """(file path, file name, concept prompt) for every image of a split of our COCO export."""
    folder = os.path.join(data_dir, COCO_FOLDERS[split])
    with open(os.path.join(folder, "_annotations.coco.json")) as f:
        coco = json.load(f)
    prompt = coco["categories"][0]["name"]
    return [(os.path.join(folder, im["file_name"]), im["file_name"], prompt) for im in coco["images"]]


def predict_split(model, data_dir, split, out_dir, save_prob_mask, device="cuda", limit=None):
    images = coco_images(data_dir, split)[:limit]
    was_training = model.training
    model.eval()
    for path, name, prompt in images:
        save_prob_mask(prob_map(model, PILImage.open(path).convert("RGB"), prompt, device),
                       os.path.join(out_dir, name))
    model.train(was_training)
    return len(images), images[0][2] if images else None
