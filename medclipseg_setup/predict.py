"""MedCLIPSeg prediction + the shared report. Run from inside the MedCLIPSeg/ repo folder, like train.py.

After training (best val Dice checkpoint of the run folder; writes preds_val/, preds_test/ and run_report.py's report):
    python ../medclipseg_setup/predict.py --config-file ../medclipseg_setup/configs/Covid19.yaml \
        --output-dir ../runs/medclipseg --seed 42

Sanity check of a released checkpoint on our val split only (no test), into its own folder:
    python ../medclipseg_setup/predict.py --config-file ../medclipseg_setup/configs/Covid19.yaml --sanity \
        --checkpoint outputs_medclipseg/Covid19/trained_models/seed666/MedCLIPSeg_unimedclip_ViT-B-16_latest.pth \
        --out_dir ../runs/medclipseg_sanity/Covid19

Predictions follow MedCLIPSeg's test protocol: mean sigmoid probability of TEST.NUM_SAMPLES (30) stochastic forward
passes, saved as probability maps (0-255) named like the images; the threshold is then picked on val by run_report.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.getcwd())  # the MedCLIPSeg repo
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, PROJECT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from datasets.dataloader import DatasetSegmentation, ValGenerator  # noqa: E402
from trainers import build_medclipseg_unimedclip  # noqa: E402
from utils.main_utils import load_cfg_from_cfg_file, read_text  # noqa: E402

from evaluate import METRICS, evaluate, select_threshold, summarize  # noqa: E402
from run_report import finalize_run, save_prob_mask  # noqa: E402

DATASETS = {"Covid19": "qata", "MosMedPlus": "mosmed"}
SPLITS = {"val": ("VAL_PATH", "Val_text.xlsx"), "test": ("TEST_PATH", "Test_text_original.xlsx")}

p = argparse.ArgumentParser()
p.add_argument("--config-file", required=True)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--output-dir", default="../runs/medclipseg")
p.add_argument("--checkpoint", help="default: the run folder's best val Dice checkpoint")
p.add_argument("--sanity", action="store_true", help="val only, no report; for checking a released checkpoint")
p.add_argument("--out_dir", help="where predictions go (default: the run folder)")
p.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = p.parse_args()
cfg = load_cfg_from_cfg_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.update(vars(args))

random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)

dataset = DATASETS[cfg.DATASET.NAME]
data_root = os.path.abspath(os.path.join(cfg.DATASET.TRAIN_PATH, ".."))
results_name = f"MedCLIPSeg_{cfg.MODEL.CLIP_MODEL}_{cfg.MODEL.BACKBONE.replace('/', '-')}"
run_dir = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}")
out_dir = cfg.out_dir or run_dir
ckpt_path = cfg.checkpoint or os.path.join(run_dir, f"{results_name}_best_dice.pth")

model = build_medclipseg_unimedclip(cfg)
ckpt = torch.load(ckpt_path, map_location=cfg.MODEL.DEVICE, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval().to(cfg.MODEL.DEVICE)
best_epoch = ckpt["epoch"] + 1
print(f"MedCLIPSeg {cfg.DATASET.NAME} | checkpoint {ckpt_path} (epoch {best_epoch})", flush=True)

for split in (["val"] if cfg.sanity else ["val", "test"]):
    path_key, text_file = SPLITS[split]
    data = DatasetSegmentation(cfg.DATASET[path_key], cfg.DATASET.NAME,
                               read_text(cfg.DATASET.TEXT_PROMPT_PATH + text_file),
                               ValGenerator(output_size=[cfg.DATASET.SIZE] * 2), image_size=cfg.DATASET.SIZE)
    loader = DataLoader(data, batch_size=32, shuffle=False, num_workers=8)
    with torch.no_grad():
        for batch in tqdm(loader, desc=split, mininterval=60):
            samples = model(image=batch["image"].to(cfg.MODEL.DEVICE), text=batch["text_prompt"],
                            num_samples=cfg.TEST.NUM_SAMPLES)          # [N, B, H, W] logits ([N, B, 1, H, W] also ok)
            probs = torch.sigmoid(samples).mean(dim=0)                 # predictive mean, as in their test.py
            if probs.ndim == 4:                                        # drop a channel dimension if there is one
                probs = probs[:, 0]
            assert probs.ndim == 3, f"unexpected prediction shape {tuple(probs.shape)}"
            probs = probs.cpu().numpy()                                # [B, H, W]
            for name, prob in zip(batch["image_name"], probs):
                save_prob_mask(prob, os.path.join(out_dir, "preds_" + split, name))
    print(f"saved {len(data)} {split} predictions", flush=True)

if cfg.sanity:
    threshold, sweep = select_threshold(dataset, os.path.join(out_dir, "preds_val"), data_root=data_root)
    results = {"checkpoint": os.path.abspath(ckpt_path), "split": "val"}
    for t in sorted({0.5, threshold}):
        s = summarize(evaluate(dataset, "val", os.path.join(out_dir, "preds_val"), t, data_root))
        results[f"threshold_{t}"] = s
        print(f"val @ threshold {t}: " + " | ".join(f"{k} {s[k]['mean']:.4f}" for k in METRICS))
    with open(os.path.join(out_dir, "sanity_val_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
else:
    finalize_run(dataset, run_dir, data_root=data_root, best_epoch=best_epoch)
