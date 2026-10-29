"""MedCLIPSeg training (MedCLIPSeg/train.py) with the project's protocol added. Run from inside the MedCLIPSeg/ repo
folder, so its own code (trainers, datasets, utils) is used unchanged:
    cd MedCLIPSeg && python ../medclipseg_setup/train.py --config-file ../medclipseg_setup/configs/Covid19.yaml \
        --output-dir ../runs/medclipseg --seed 42 [--resume]

Same model, loss (0.5 Dice + 0.5 BCE + 0.1 CLIP loss), optimizer, scheduler, augmentation and val Dice as theirs.
Changes:
  - early stopping: stop after --patience (20) epochs without a new best val Dice (max epochs from the config, 50)
  - resume works: theirs looks for the latest checkpoint outside the seed folder it saves to (so --resume never finds
    it) and resets the best val Dice; here both live in the run folder and the best Dice / epoch are restored
  - time budget: stops after the epoch that would run past --time_budget_hours (the gpu partition has a 12 h limit),
    so the slurm job can resubmit itself with --resume; TRAINING_DONE is written when training is really finished
  - history.csv (epoch, train_loss, train_dice, val_loss, val_dice) and run_config.json for run_report.py
Run folder: <output-dir>/<dataset name>/trained_models/seed<seed>/
"""
import argparse
import csv
import logging
import os
import random
import sys
import time
from statistics import mean

sys.path.insert(0, os.getcwd())  # the MedCLIPSeg repo
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, PROJECT)

import monai  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn.modules.loss import BCEWithLogitsLoss  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms  # noqa: E402
from tqdm import tqdm  # noqa: E402

from datasets.dataloader import DatasetSegmentation, RandomGenerator, ValGenerator  # noqa: E402
from trainers import build_medclipseg_unimedclip  # noqa: E402
from utils.main_utils import load_cfg_from_cfg_file, read_text  # noqa: E402

from run_report import write_run_config  # noqa: E402

DATASETS = {"Covid19": "qata", "MosMedPlus": "mosmed"}


def get_cfg():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="../runs/medclipseg")
    p.add_argument("--patience", type=int, default=20, help="stop after this many epochs without a new best val Dice")
    p.add_argument("--time_budget_hours", type=float, default=11.0, help="stop (to resume) before this much time")
    p.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="config overrides: KEY VALUE ...")
    args = p.parse_args()
    cfg = load_cfg_from_cfg_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.update(vars(args))
    return cfg


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_loss(logits, masks, ce_loss, dice_loss, cfg):
    return cfg.TRAIN.DICE_WEIGHT * dice_loss(logits, masks) + cfg.TRAIN.CE_WEIGHT * ce_loss(logits, masks.float())


def hard_dice(logits, masks):
    """Per-image Dice at threshold 0.5, as in their validation."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    if preds.ndim == 3:
        preds = preds.unsqueeze(1)
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    inter = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return ((2.0 * inter + 1e-7) / (union + 1e-7)).cpu().tolist()


def main():
    cfg = get_cfg()
    start_time = time.time()
    results_name = f"MedCLIPSeg_{cfg.MODEL.CLIP_MODEL}_{cfg.MODEL.BACKBONE.replace('/', '-')}"
    run_dir = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}")
    os.makedirs(run_dir, exist_ok=True)
    latest_path = os.path.join(run_dir, f"{results_name}_latest.pth")
    best_path = os.path.join(run_dir, f"{results_name}_best_dice.pth")
    history_path = os.path.join(run_dir, "history.csv")

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.FileHandler(os.path.join(run_dir, "log.txt"), encoding="utf-8"),
                                  logging.StreamHandler()])
    log = logging.getLogger()
    log.info(cfg)
    set_random_seed(cfg.seed)

    def worker_init_fn(worker_id):
        random.seed(cfg.seed + worker_id)
        np.random.seed(cfg.seed + worker_id)

    size = [cfg.DATASET.SIZE, cfg.DATASET.SIZE]
    train_set = DatasetSegmentation(cfg.DATASET.TRAIN_PATH, cfg.DATASET.NAME,
                                    read_text(cfg.DATASET.TEXT_PROMPT_PATH + "Train_text.xlsx"),
                                    transforms.Compose([RandomGenerator(output_size=size)]), image_size=cfg.DATASET.SIZE)
    val_set = DatasetSegmentation(cfg.DATASET.VAL_PATH, cfg.DATASET.NAME,
                                  read_text(cfg.DATASET.TEXT_PROMPT_PATH + "Val_text.xlsx"),
                                  ValGenerator(output_size=size), image_size=cfg.DATASET.SIZE)
    train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=8,
                              worker_init_fn=worker_init_fn, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=8,
                            worker_init_fn=worker_init_fn, pin_memory=True)
    log.info(f"train: {len(train_set)} images | val: {len(val_set)} images")

    model = build_medclipseg_unimedclip(cfg)
    log.info(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    ce_loss = BCEWithLogitsLoss()
    dice_loss = monai.losses.DiceLoss(include_background=False, sigmoid=True, reduction="mean")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LEARNING_RATE)
    num_epochs = cfg.TRAIN.NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

    start_epoch, best_dice, best_epoch = 0, 0.0, 0
    if cfg.resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best_dice, best_epoch = ckpt["epoch"] + 1, ckpt["best_dice"], ckpt["best_epoch"]
        log.info(f"resumed after epoch {start_epoch} (best val Dice {best_dice:.4f} in epoch {best_epoch})")
    else:
        with open(history_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_dice", "val_loss", "val_dice"])

    write_run_config(run_dir, {
        "model": "MedCLIPSeg", "dataset": DATASETS[cfg.DATASET.NAME], "config_file": os.path.abspath(cfg.config_file),
        "data_dir": os.path.abspath(os.path.join(cfg.DATASET.TRAIN_PATH, "..")),
        "n_train": len(train_set), "n_val": len(val_set),
        "preprocessing": "image read as 3-channel uint8 and resized to {0}x{0} (cv2 bilinear, then PIL bicubic), "
                         "mask 0/255 resized nearest and >= 127 = lesion (exported masks are 0/255 with > 0 = lesion); "
                         "CLIP normalization (ImageNet mean/std); train augmentation: random rotation +-20 deg "
                         "(p=0.5)".format(cfg.DATASET.SIZE),
        "prompts": "MedCLIPSeg's wording of the LViT text annotation of each image (Prompts_Folder/*.xlsx; for QaTa "
                   "identical to their released prompts), encoded by the BiomedBERT text encoder of UniMedCLIP",
        "pretrained": "UniMedCLIP ViT-B/16 (checkpoints/unimed_clip_vit_b16.pt from TahaKoleilat/MedCLIPSeg) + "
                      "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        "training": {"loss": "{} Dice + {} BCE + {} CLIP loss".format(cfg.TRAIN.DICE_WEIGHT, cfg.TRAIN.CE_WEIGHT,
                                                                     cfg.TRAIN.CLIP_WEIGHT),
                     "optimizer": "Adam", "lr": cfg.TRAIN.LEARNING_RATE,
                     "scheduler": f"CosineAnnealingLR(T_max={num_epochs}, eta_min=1e-4), stepped per epoch",
                     "batch_size": cfg.TRAIN.BATCH_SIZE, "max_epochs": num_epochs,
                     "early_stopping": f"stop after {cfg.patience} epochs without a new best val Dice",
                     "checkpoint_rule": "best val Dice (hard, threshold 0.5, 1 forward pass)"},
        "inference": f"mean of {cfg.TEST.NUM_SAMPLES} stochastic forward passes (MedCLIPSeg's test protocol)",
        "seed": cfg.seed, "checkpoint": os.path.abspath(best_path)})

    model.train().to(cfg.MODEL.DEVICE)
    budget = cfg.time_budget_hours * 3600
    longest_epoch = 0.0
    finished = True
    for epoch in range(start_epoch, num_epochs):
        if longest_epoch and time.time() - start_time + 1.1 * longest_epoch > budget:
            log.info(f"time budget of {cfg.time_budget_hours} h reached after epoch {epoch}: resubmit with --resume")
            finished = False
            break
        epoch_start = time.time()
        losses, dices = [], []
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}", mininterval=60):
            masks = batch["ground_truth_mask"].to(cfg.MODEL.DEVICE)
            seg_logits, clip_loss = model(image=batch["image"].to(cfg.MODEL.DEVICE), text=batch["text_prompt"])
            loss = calc_loss(seg_logits, masks, ce_loss, dice_loss, cfg) + cfg.TRAIN.CLIP_WEIGHT * clip_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            dices.extend(hard_dice(seg_logits.detach(), masks))
        scheduler.step()

        model.eval()
        val_losses, val_dices = [], []
        with torch.no_grad():
            for batch in val_loader:
                masks = batch["ground_truth_mask"].to(cfg.MODEL.DEVICE)
                logits = model(batch["image"].to(cfg.MODEL.DEVICE), text=batch["text_prompt"], num_samples=1)[0]
                val_losses.append(calc_loss(logits, masks, ce_loss, dice_loss, cfg).item())
                val_dices.extend(hard_dice(logits, masks))
        model.train()
        train_loss, train_dice, val_loss, val_dice = mean(losses), mean(dices), mean(val_losses), mean(val_dices)
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, train_dice, val_loss, val_dice])
        log.info(f"EPOCH {epoch + 1} | train loss {train_loss:.4f} Dice {train_dice:.4f} | "
                 f"val loss {val_loss:.4f} Dice {val_dice:.4f}")

        if val_dice > best_dice:
            log.info(f"\t new best val Dice: {best_dice:.4f} -> {val_dice:.4f}")
            best_dice, best_epoch = val_dice, epoch + 1
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_dice": best_dice}, best_path)
        torch.save({"model": model.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "best_dice": best_dice, "best_epoch": best_epoch}, latest_path)

        longest_epoch = max(longest_epoch, time.time() - epoch_start)
        early_stopping_count = epoch + 1 - best_epoch
        log.info(f"\t early_stopping_count: {early_stopping_count}/{cfg.patience}")
        if early_stopping_count > cfg.patience:
            log.info("\t early_stopping!")
            break

    if finished:
        with open(os.path.join(run_dir, "TRAINING_DONE"), "w") as f:
            f.write(f"best val Dice {best_dice:.4f} in epoch {best_epoch}\n")
        log.info(f"training done: best val Dice {best_dice:.4f} in epoch {best_epoch}")


if __name__ == "__main__":
    main()
