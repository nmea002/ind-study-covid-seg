"""Finish a training run the same way for every model (U-Net, LViT, MedCLIPSeg, MedSAM3).

A model's training script only has to leave this in its run folder:
    preds_val/<image name>.png    probability map (0-255 = 0-1) from the best checkpoint, one per val image
    preds_test/<image name>.png   the same for every test image
    history.csv                   one row per epoch: epoch, train_loss, train_dice, val_loss, val_dice
    run_config.json               what the run used (preprocessing, prompts, settings, checkpoint, seed)
(save_prob_mask() and write_run_config() below help with the first and last.)

finalize_run() then:
  1. picks the threshold with the best mean val Dice   -> the only setting chosen after training, on val only
  2. evaluates val and test once at that threshold      -> metrics.json, per_image_val.csv, per_image_test.csv
  3. plots the training curves                         -> curves.png
  4. saves image | ground truth | prediction figures   -> examples/ (the same test images for every model)
  5. adds the threshold and the metrics to run_config.json

Usage (re-run on an existing run folder):
    python run_report.py --dataset qata --run_dir <run folder> [--data_root <LViT-layout dataset folder>]
"""
import argparse
import json
import os
import subprocess

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evaluate import METRICS, evaluate, gt_path, image_path, load_split, read_mask, select_threshold, summarize  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
N_EXAMPLES = 5
EXAMPLE_SEED = 0
DATASET_NAMES = {"qata": "QaTa-COV19-v2", "mosmed": "MosMedData+"}


def save_prob_mask(prob, path):
    """Save a 2D probability map (floats in 0-1) as an 8-bit PNG (0-255)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.clip(np.round(np.asarray(prob, dtype=np.float32) * 255), 0, 255).astype(np.uint8))


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_config(run_dir, config):
    """run_config.json; the split files and git commit are added automatically."""
    config = {**config, "split_csv": f"splits/{config['dataset']}_split.csv", "git_commit": git_commit()}
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)


def example_images(dataset):
    """The same N_EXAMPLES test images for every model (seeded random pick)."""
    test = load_split(dataset, "test")
    return [r for _, r in test.sample(N_EXAMPLES, random_state=EXAMPLE_SEED).iterrows()]


def plot_curves(history_csv, out_png, title, best_epoch=None):
    h = pd.read_csv(history_csv)
    best_epoch = best_epoch or int(h.loc[h["val_dice"].idxmax(), "epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["loss", "dice"]):
        ax.plot(h["epoch"], h["train_" + metric], label="train")
        ax.plot(h["epoch"], h["val_" + metric], label="val")
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label=f"checkpoint (epoch {best_epoch})")
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric.capitalize())
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def save_examples(dataset, run_dir, threshold, per_image_test, data_root, title):
    out_dir = os.path.join(run_dir, "examples")
    os.makedirs(out_dir, exist_ok=True)
    scores = per_image_test.set_index("Image")
    for r in example_images(dataset):
        pred_file = os.path.join(run_dir, "preds_test", r["Image"])
        if not os.path.exists(pred_file):  # debug runs on a few images
            continue
        image = cv2.cvtColor(cv2.imread(image_path(dataset, r, data_root)), cv2.COLOR_BGR2RGB)
        gt = read_mask(gt_path(dataset, r, data_root)) > 0
        image = cv2.resize(image, (gt.shape[1], gt.shape[0]))
        prob = read_mask(pred_file)
        prob = cv2.resize(prob, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        pred = prob.astype(np.float32) / 255 > threshold

        overlay = image.copy()
        overlay[gt & ~pred] = (0.5 * overlay[gt & ~pred] + [0, 0, 127]).astype(np.uint8)    # missed: blue
        overlay[pred & ~gt] = (0.5 * overlay[pred & ~gt] + [127, 0, 0]).astype(np.uint8)    # false alarm: red
        overlay[pred & gt] = (0.5 * overlay[pred & gt] + [0, 127, 0]).astype(np.uint8)      # hit: green

        s = scores.loc[r["Image"]]
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        for ax, img, name in zip(axes, [image, gt, pred, overlay],
                                 ["image", "ground truth", "prediction", "overlay (green hit, red false, blue missed)"]):
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(name, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"{title} | {r['Image']} | Dice {s['dice']:.3f}  IoU {s['iou']:.3f}  "
                     f"HD95 {s['hd95']:.1f}  ASSD {s['assd']:.1f}\nprompt: {r['text']}", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, os.path.splitext(r["Image"])[0] + ".png"), dpi=110)
        plt.close(fig)


def finalize_run(dataset, run_dir, data_root=None, require_all=True, best_epoch=None):
    """best_epoch: epoch of the checkpoint that made the predictions (default: run_config's, else best val Dice)."""
    config_path = os.path.join(run_dir, "run_config.json")
    config = json.load(open(config_path)) if os.path.exists(config_path) else {}
    best_epoch = best_epoch or config.get("best_epoch")
    title = f"{config.get('model', 'model')} on {DATASET_NAMES[dataset]}"

    threshold, sweep = select_threshold(dataset, os.path.join(run_dir, "preds_val"), data_root=data_root,
                                        require_all=require_all)
    results = {"threshold": threshold, "threshold_selected_by": "best mean val Dice", "val_dice_by_threshold": sweep}
    for split in ["val", "test"]:
        per_image = evaluate(dataset, split, os.path.join(run_dir, "preds_" + split), threshold, data_root, require_all)
        per_image.to_csv(os.path.join(run_dir, f"per_image_{split}.csv"), index=False)
        results[split] = summarize(per_image)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    history = os.path.join(run_dir, "history.csv")
    if os.path.exists(history):
        plot_curves(history, os.path.join(run_dir, "curves.png"), title, best_epoch)
    save_examples(dataset, run_dir, threshold, per_image, data_root, title)

    if config:
        config.update({"best_epoch": best_epoch, "threshold": threshold, "metrics": "metrics.json"})
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    print(f"\n{title} | threshold {threshold} (best val Dice)")
    print("| split | Dice | IoU | HD95 (px) | ASSD (px) |")
    print("|---|---|---|---|---|")
    for split in ["val", "test"]:
        s = results[split]
        print(f"| {split} | " + " | ".join(f"{s[k]['mean']:.4f} +- {s[k]['std']:.4f}" for k in METRICS) + " |")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["qata", "mosmed"])
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_root", help="LViT-layout copy of the dataset (default: the raw data in LViT/datasets)")
    a = p.parse_args()
    finalize_run(a.dataset, a.run_dir, a.data_root)
