"""MedSAM3 prediction + the shared report. Run from inside the MedSAM3/ repo folder:
    cd MedSAM3 && python ../medsam3_setup/predict.py --dataset qata                      # zero-shot: released LoRA
    cd MedSAM3 && python ../medsam3_setup/predict.py --dataset qata --weights <best_lora_weights.pt> --run_dir <dir>

Text/concept prompt only (the category name of our COCO export, e.g. "covid-19 infection"); no boxes, points or
ground truth are used to prompt. Instances are combined into one map as described in common.py.
Ground truth for the report comes from the MedCLIPSeg export (same frozen split, same binarized masks).
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())  # the MedSAM3 repo
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, PROJECT)
sys.path.insert(2, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from common import (DETECTION_THRESHOLD, NMS_IOU, RELEASED_LORA, RESOLUTION, build_inference_model,  # noqa: E402
                    predict_split)
from run_report import finalize_run, save_prob_mask, write_run_config  # noqa: E402

GT_ROOTS = {"qata": "Covid19", "mosmed": "MosMedPlus"}


def gt_root(dataset):
    return os.path.join(PROJECT, "model_data", "medclipseg", GT_ROOTS[dataset])


def describe(dataset, weights, how):
    return {
        "model": "MedSAM3", "dataset": dataset, "variant": how,
        "preprocessing": f"image read as RGB (PIL), resized to {RESOLUTION}x{RESOLUTION}, normalized to [-1, 1] "
                         "(SAM3 transforms); masks from the frozen split, any value > 0 = lesion",
        "prompts": "one text/concept prompt per dataset (category name in model_data/medsam3/<dataset>/*/"
                   "_annotations.coco.json); no box/point/ground-truth prompts",
        "instance_rule": f"detections with score > {DETECTION_THRESHOLD}, NMS at IoU {NMS_IOU} on their boxes, pixel "
                         "probability = max mask probability over the kept detections (0 if none)",
        "pretrained": "facebook/sam3 + MedSAM3 v1 LoRA (lal-Joey/MedSAM3_v1), LoRA rank 16 / alpha 32 on all "
                      "components (MedSAM3/configs/full_lora_config.yaml)",
        "seed": 42, "checkpoint": os.path.abspath(weights)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["qata", "mosmed"])
    p.add_argument("--config", help="default: ../medsam3_setup/configs/<dataset>.yaml")
    p.add_argument("--weights", default=RELEASED_LORA, help="LoRA weights (default: released MedSAM3 v1)")
    p.add_argument("--run_dir", help="default: ../runs/medsam3/<dataset>/zeroshot")
    p.add_argument("--limit", type=int, default=None, help="debug: only the first N images per split")
    a = p.parse_args()
    config = a.config or os.path.join(PROJECT, "medsam3_setup", "configs", f"{a.dataset}.yaml")
    run_dir = a.run_dir or os.path.join(PROJECT, "runs", "medsam3", a.dataset, "zeroshot")
    data_dir = os.path.join(PROJECT, "model_data", "medsam3", a.dataset)
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(42)

    zero_shot = os.path.basename(a.weights) == RELEASED_LORA
    if zero_shot:
        write_run_config(run_dir, describe(a.dataset, a.weights, "zero-shot (released MedSAM3 v1 LoRA, no fine-tuning)"))
    model = build_inference_model(config, a.weights)
    for split in ["val", "test"]:
        n, prompt = predict_split(model, data_dir, split, os.path.join(run_dir, "preds_" + split), save_prob_mask,
                                  limit=a.limit)
        print(f"saved {n} {split} predictions (prompt: {prompt!r})", flush=True)
    finalize_run(a.dataset, run_dir, data_root=gt_root(a.dataset), require_all=a.limit is None)
