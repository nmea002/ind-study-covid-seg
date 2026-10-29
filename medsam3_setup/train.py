"""MedSAM3 LoRA fine-tuning with the project's protocol. Run from inside the MedSAM3/ repo folder:
    cd MedSAM3 && python ../medsam3_setup/train.py --dataset qata

Model, LoRA setup, losses (SAM3's matcher + box/class/mask losses), optimizer and the training step are those of
MedSAM3/train_sam3_lora_native.py (SAM3TrainerNative), on our COCO export of the frozen split. Changes:
  - starts from the released MedSAM3 v1 LoRA weights (their script starts LoRA from scratch)
  - checkpoint selection and early stopping on val Dice (their script keeps the lowest val loss): after every epoch
    the val images are predicted exactly as in predict.py (text prompt only) and scored at threshold 0.5;
    stop after --patience (20) epochs without a new best, at most num_epochs (50) from the config
  - time budget: stops before an epoch that would run past --time_budget_hours, or mid-epoch once it is used up
    (that partial epoch is still validated), so the best checkpoint can still be predicted and reported inside the
    job (the run then stopped on time, not by early stopping: see stop_reason.txt)
  - history.csv, run_config.json, then predict val + test with the best weights and run_report.finalize_run
Run folder: ../runs/medsam3/<dataset>/finetune/
"""
import argparse
import csv
import os
import random
import sys
import time

sys.path.insert(0, os.getcwd())  # the MedSAM3 repo
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, PROJECT)
sys.path.insert(2, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common import RELEASED_LORA, load_lora_checked, predict_split  # noqa: E402
from predict import describe, gt_root  # noqa: E402
from train_sam3_lora_native import (CORE_LOSS_KEY, COCOSegmentDataset, SAM3Output, SAM3TrainerNative,  # noqa: E402
                                    collate_fn_api)
from lora_layers import save_lora_weights  # noqa: E402

from evaluate import select_threshold  # noqa: E402
from run_report import finalize_run, save_prob_mask, write_run_config  # noqa: E402


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            setattr(obj, field, to_device(getattr(obj, field), device))
    return obj


def train_step(trainer, input_batch):
    """One optimization step, as in SAM3TrainerNative.train()."""
    model = trainer.model
    outputs_list = model(input_batch)
    find_targets = [trainer._unwrapped_model.back_convert(t) for t in input_batch.find_targets]
    for targets in find_targets:
        for k, v in targets.items():
            if isinstance(v, torch.Tensor):
                targets[k] = v.to(trainer.device)
    with SAM3Output.iteration_mode(outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE) as outputs_iter:
        for stage_outputs, stage_targets in zip(outputs_iter, find_targets):
            for outputs in stage_outputs:
                outputs["indices"] = trainer.matcher(outputs, stage_targets)
                for aux in outputs.get("aux_outputs", []):
                    aux["indices"] = trainer.matcher(aux, stage_targets)
    loss = trainer.loss_wrapper(outputs_list, find_targets)[CORE_LOSS_KEY]
    trainer.optimizer.zero_grad()
    loss.backward()
    trainer.optimizer.step()
    return loss.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["qata", "mosmed"])
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--time_budget_hours", type=float, default=9.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="debug: only the first N train/val images")
    a = p.parse_args()
    config = os.path.join(PROJECT, "medsam3_setup", "configs", f"{a.dataset}.yaml")
    data_dir = os.path.join(PROJECT, "model_data", "medsam3", a.dataset)
    run_dir = os.path.join(PROJECT, "runs", "medsam3", a.dataset, "finetune")
    os.makedirs(run_dir, exist_ok=True)
    best_path, history_path = os.path.join(run_dir, "best_lora_weights.pt"), os.path.join(run_dir, "history.csv")
    val_dir = os.path.join(run_dir, "preds_val")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    start = time.time()

    trainer = SAM3TrainerNative(config)
    load_lora_checked(trainer.model, RELEASED_LORA)
    cfg = trainer.config
    num_epochs = cfg["training"]["num_epochs"]

    train_ds = COCOSegmentDataset(data_dir=data_dir, split="train")
    if a.limit:
        train_ds.image_ids = train_ds.image_ids[:a.limit]
    loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
                        collate_fn=lambda b: collate_fn_api(b, dict_key="input", with_seg_masks=True),
                        num_workers=cfg["training"]["num_workers"], pin_memory=True)

    write_run_config(run_dir, {
        **describe(a.dataset, best_path, "LoRA fine-tuned from the released MedSAM3 v1 LoRA"),
        "n_train": len(train_ds),
        "training": {"loss": "SAM3 losses as in MedSAM3/train_sam3_lora_native.py (Hungarian matching; box L1 + GIoU,"
                             " IABCE class/presence, focal + Dice mask)",
                     "optimizer": "AdamW", "lr": cfg["training"]["learning_rate"],
                     "weight_decay": cfg["training"]["weight_decay"], "batch_size": cfg["training"]["batch_size"],
                     "max_epochs": num_epochs, "lora_rank": cfg["lora"]["rank"],
                     "early_stopping": f"stop after {a.patience} epochs without a new best val Dice",
                     "checkpoint_rule": "best val Dice (text-prompted val predictions, threshold 0.5)",
                     "time_budget_hours": a.time_budget_hours},
        "seed": a.seed})
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_dice", "val_loss", "val_dice"])

    def val_dice():
        predict_split(trainer.model, data_dir, "val", val_dir, save_prob_mask, limit=a.limit)
        _, sweep = select_threshold(a.dataset, val_dir, thresholds=[0.5], data_root=gt_root(a.dataset),
                                    require_all=a.limit is None)
        return sweep[0.5]

    best_dice = val_dice()  # epoch 0 = the released weights
    best_epoch = 0
    save_lora_weights(trainer.model, best_path)
    print(f"epoch 0 (released MedSAM3 LoRA): val Dice {best_dice:.4f}", flush=True)
    longest = 0.0
    stop_reason = "max epochs"
    for epoch in range(1, num_epochs + 1):
        if longest and time.time() - start + 1.1 * longest > a.time_budget_hours * 3600:
            stop_reason = f"time budget ({a.time_budget_hours} h) after epoch {epoch - 1}"
            break
        t0 = time.time()
        trainer.model.train()
        losses = []
        for b in tqdm(loader, desc=f"epoch {epoch}", mininterval=60):
            losses.append(train_step(trainer, to_device(b["input"], trainer.device)))
            if time.time() - start > a.time_budget_hours * 3600:  # also inside an epoch: a slow GPU must not
                break                                              # run the job out before predict + report
        dice = val_dice()
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, float(np.mean(losses)), "", "", dice])
        print(f"epoch {epoch}: train loss {np.mean(losses):.4f} | val Dice {dice:.4f}", flush=True)
        if dice > best_dice:
            print(f"\t new best val Dice: {best_dice:.4f} -> {dice:.4f}", flush=True)
            best_dice, best_epoch = dice, epoch
            save_lora_weights(trainer.model, best_path)
        longest = max(longest, time.time() - t0)
        print(f"\t early_stopping_count: {epoch - best_epoch}/{a.patience}", flush=True)
        if len(losses) < len(loader):
            stop_reason = (f"time budget ({a.time_budget_hours} h) during epoch {epoch} "
                           f"({len(losses)}/{len(loader)} steps)")
            break
        if epoch - best_epoch > a.patience:
            stop_reason = "early stopping"
            break
    print(f"stopped: {stop_reason} | best val Dice {best_dice:.4f} in epoch {best_epoch}", flush=True)

    load_lora_checked(trainer.model, best_path)
    for split in ["val", "test"]:
        n, _ = predict_split(trainer.model, data_dir, split, os.path.join(run_dir, "preds_" + split), save_prob_mask,
                             limit=a.limit)
        print(f"saved {n} {split} predictions", flush=True)
    finalize_run(a.dataset, run_dir, data_root=gt_root(a.dataset), require_all=a.limit is None, best_epoch=best_epoch)
    with open(os.path.join(run_dir, "stop_reason.txt"), "w") as f:
        f.write(f"{stop_reason}; best val Dice {best_dice:.4f} in epoch {best_epoch}\n")


if __name__ == "__main__":
    main()
