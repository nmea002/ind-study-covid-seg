"""Shared evaluation for every model (U-Net, LViT, MedCLIPSeg, MedSAM3): Dice, foreground IoU, HD95 and ASSD.

Each model only has to save one predicted mask per image as a PNG named like the image in the split CSV
(e.g. covid_1.png, Jun_coronacases_case8_12.png), either binary (0/255) or a probability map (0-255 = 0-1).
This script does the rest the same way for all models, so the numbers are comparable:
  - ground truth: the raw masks (QaTa 224x224, MosMed 512x512), foreground = any value > 0 (same as LViT's loader).
    Read from the raw download by default, or with --data_root from an LViT-layout copy of the dataset
    (<data_root>/{Train,Val,Test}_Folder/labelcol/, e.g. on the server, where only the unzipped copy exists)
  - predictions are resized to the ground-truth size (bilinear) and thresholded at --threshold (default 0.5);
    select_threshold() picks the threshold with the best mean Dice on val (use that one for test)
  - metrics are computed per image at the ground-truth resolution, then averaged over the images (mean, std)
  - HD95 / ASSD are in pixels (the PNGs have no physical spacing), over the mask boundaries, computed like
    medpy.metric.binary.hd95 / assd (boundary = mask minus its 4-connected erosion)
  - empty masks: prediction and ground truth both empty -> Dice = IoU = 1, HD95 = ASSD = 0;
    only one of them empty -> Dice = IoU = 0, HD95 = ASSD = image diagonal (the largest possible distance)

Usage:
    python evaluate.py --dataset qata --split val --pred_dir runs/unet_qata/preds_val
    python evaluate.py --dataset mosmed --split test --pred_dir <dir> --out_dir <dir>   # writes per-image CSV + summary

    from evaluate import segmentation_metrics   # per-image metrics on two boolean arrays
    from evaluate import select_threshold, evaluate, summarize
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(ROOT, "LViT", "datasets")
QATA_RAW = os.path.join(DATASETS, "archive", "QaTa-COV19", "QaTa-COV19-v2")
METRICS = ["dice", "iou", "hd95", "assd"]
LVIT_FOLDERS = {"train": "Train_Folder", "val": "Val_Folder", "test": "Test_Folder"}


def load_split(dataset, split):
    df = pd.read_csv(os.path.join(ROOT, "splits", f"{dataset}_split.csv"))
    return df[df["split"] == split]


def gt_path(dataset, row, data_root=None):
    if data_root:
        return os.path.join(data_root, LVIT_FOLDERS[row["split"]], "labelcol", row["Mask"])
    if dataset == "qata":
        raw_set = "Test Set" if row["split"] == "test" else "Train Set"
        return os.path.join(QATA_RAW, raw_set, "Ground-truths", row["Mask"])
    return os.path.join(DATASETS, "MosMedData", "masks", row["Mask"])


def image_path(dataset, row, data_root=None):
    if data_root:
        return os.path.join(data_root, LVIT_FOLDERS[row["split"]], "img", row["Image"])
    if dataset == "qata":
        raw_set = "Test Set" if row["split"] == "test" else "Train Set"
        return os.path.join(QATA_RAW, raw_set, "Images", row["Image"])
    return os.path.join(DATASETS, "MosMedData", "frames", row["Image"])


def read_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m


def _boundary(mask):
    return mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(2, 1))


def _surface_distances(a, b):
    """Distance from every boundary pixel of a to the nearest boundary pixel of b."""
    return ndimage.distance_transform_edt(~_boundary(b))[_boundary(a)]


def segmentation_metrics(pred, gt):
    """Dice, foreground IoU, HD95 and ASSD (pixels) for one pair of 2D boolean masks of the same shape."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    assert pred.shape == gt.shape, f"shape mismatch {pred.shape} vs {gt.shape}"
    if not pred.any() and not gt.any():
        return {"dice": 1.0, "iou": 1.0, "hd95": 0.0, "assd": 0.0}
    if not pred.any() or not gt.any():
        diagonal = float(np.hypot(*gt.shape))
        return {"dice": 0.0, "iou": 0.0, "hd95": diagonal, "assd": diagonal}
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    distances = np.concatenate([_surface_distances(pred, gt), _surface_distances(gt, pred)])
    return {
        "dice": float(2 * inter / (pred.sum() + gt.sum())),
        "iou": float(inter / union),
        "hd95": float(np.percentile(distances, 95)),
        "assd": float(distances.mean()),
    }


def _pairs(dataset, split, pred_dir, data_root=None, require_all=True):
    """Yield (image name, prediction scaled to 0-1 at ground-truth size, boolean ground truth) for one split.
    require_all=False only evaluates the images that have a prediction (for quick debug runs)."""
    df = load_split(dataset, split)
    missing = [f for f in df["Image"] if not os.path.exists(os.path.join(pred_dir, f))]
    if require_all:
        assert not missing, f"{len(missing)} of {len(df)} predictions missing in {pred_dir}, e.g. {missing[:3]}"
    df = df[~df["Image"].isin(missing)]
    assert len(df), f"no predictions for the {dataset} {split} images in {pred_dir}"
    for _, r in df.iterrows():
        gt = read_mask(gt_path(dataset, r, data_root)) > 0
        pred = read_mask(os.path.join(pred_dir, r["Image"]))
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        yield r["Image"], pred.astype(np.float32) / 255, gt


def evaluate(dataset, split, pred_dir, threshold=0.5, data_root=None, require_all=True):
    """Per-image metrics for every image of one split; every image must have a prediction."""
    rows = []
    for name, prob, gt in _pairs(dataset, split, pred_dir, data_root, require_all):
        pred = prob > threshold
        m = segmentation_metrics(pred, gt)
        m["empty_pred"], m["empty_gt"] = not pred.any(), not gt.any()
        rows.append({"Image": name, **m})
    return pd.DataFrame(rows)


def select_threshold(dataset, pred_dir, thresholds=None, data_root=None, require_all=True):
    """Threshold with the best mean Dice on the val split. Returns (best threshold, {threshold: mean val Dice})."""
    thresholds = thresholds if thresholds is not None else [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
    dice_sums = np.zeros(len(thresholds))
    n = 0
    for _, prob, gt in _pairs(dataset, "val", pred_dir, data_root, require_all):
        for i, t in enumerate(thresholds):
            pred = prob > t
            denom = pred.sum() + gt.sum()
            dice_sums[i] += 1.0 if denom == 0 else 2 * np.logical_and(pred, gt).sum() / denom
        n += 1
    sweep = {float(t): float(d / n) for t, d in zip(thresholds, dice_sums)}
    return max(sweep, key=sweep.get), sweep


def summarize(per_image):
    s = {k: {"mean": float(per_image[k].mean()), "std": float(per_image[k].std())} for k in METRICS}
    s["n_images"] = len(per_image)
    s["n_empty_pred"] = int(per_image["empty_pred"].sum())
    s["n_empty_gt"] = int(per_image["empty_gt"].sum())
    return s


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["qata", "mosmed"])
    p.add_argument("--split", required=True, choices=["val", "test"])
    p.add_argument("--pred_dir", required=True, help="folder with one predicted mask PNG per image")
    p.add_argument("--threshold", type=float, default=0.5, help="on predictions scaled to 0-1; pick it on val")
    p.add_argument("--data_root", help="LViT-layout copy of the dataset to read the ground truth from (default: raw)")
    p.add_argument("--out_dir", help="where to write per_image_<split>.csv and metrics_<split>.json")
    a = p.parse_args()

    per_image = evaluate(a.dataset, a.split, a.pred_dir, a.threshold, a.data_root)
    summary = {"dataset": a.dataset, "split": a.split, "pred_dir": os.path.abspath(a.pred_dir),
               "threshold": a.threshold, **summarize(per_image)}
    for k in METRICS:
        print(f"{k:5s}: {summary[k]['mean']:.4f} +- {summary[k]['std']:.4f}")
    print(f"{summary['n_images']} images, {summary['n_empty_pred']} empty predictions, "
          f"{summary['n_empty_gt']} empty ground truths")
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        per_image.to_csv(os.path.join(a.out_dir, f"per_image_{a.split}.csv"), index=False)
        with open(os.path.join(a.out_dir, f"metrics_{a.split}.json"), "w") as f:
            json.dump(summary, f, indent=2)
