"""Train + test the base U-Net (LViT/nets/UNet.py) on QaTa-COV19-v2 or MosMedData+, for comparison with LViT.

Everything except the model is kept the same as the LViT runs, using the LViT repo's own code where possible:
  - data: the same zips / folders (Train_Folder, Val_Folder, Test_Folder), so the same full split
  - augmentation: Load_Dataset.RandomGenerator / ValGenerator, images read and resized to 224 the same way
  - loss: utils.WeightedDiceBCE(0.5, 0.5); Adam lr 3e-4; cosine warm restarts (T_0=10, eta_min=1e-4) stepped per epoch
  - batch size 2, seed 42; best model = best val Dice, only saved after epoch 5 (same rule as train_model.py)
  - max 50 epochs, early stop after 20 epochs without a new best val Dice (same as LViT's Config.py)
The U-Net takes no text, so the BERT text embedding is skipped (bert_embedding is stubbed out to import Load_Dataset).

After training, the best checkpoint predicts every val and test image (probability maps in preds_val/, preds_test/),
and run_report.finalize_run picks the threshold on val and reports Dice / IoU / HD95 / ASSD with evaluate.py,
plus curves.png (from history.csv) and examples/. Settings are recorded in run_config.json.

Usage:
    python train_unet.py --task Covid19       # QaTa-COV19-v2
    python train_unet.py --task MosMedPlus --lvit_dir <unzipped LViT folder>   # MosMedData+
    python train_unet.py --task MosMedPlus --lvit_dir <...> --test_only <session folder>   # predict + report only
"""
import argparse
import csv
import os
import random
import sys
import time
import types

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

p = argparse.ArgumentParser()
p.add_argument("--task", required=True, choices=["Covid19", "MosMedPlus"])
p.add_argument("--lvit_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "LViT"),
               help="LViT folder (code + datasets/<task>)")
p.add_argument("--out_dir", default=None, help="where sessions are saved (default <lvit_dir>/UNet_runs/<task>)")
p.add_argument("--epochs", type=int, default=50, help="max epochs")
p.add_argument("--patience", type=int, default=20, help="stop after this many epochs without a new best val Dice")
p.add_argument("--batch_size", type=int, default=2)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--img_size", type=int, default=224)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--workers", type=int, default=2)
p.add_argument("--print_frequency", type=int, default=50)
p.add_argument("--limit", type=int, default=0, help="debug: only use the first N images of each folder")
p.add_argument("--test_only", default=None, help="session folder to predict + report on (skips training)")
args = p.parse_args()

# reuse the LViT repo's code; Load_Dataset imports bert_embedding at the top, which the U-Net doesn't need
sys.path.insert(0, args.lvit_dir)
sys.modules["bert_embedding"] = types.SimpleNamespace(BertEmbedding=None)
from Load_Dataset import RandomGenerator, ValGenerator, correct_dims  # noqa: E402
from nets.UNet import UNet  # noqa: E402
from utils import WeightedDiceBCE, iou_on_batch  # noqa: E402

from run_report import finalize_run, save_prob_mask, write_run_config  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = os.path.join(args.lvit_dir, "datasets", args.task)
out_dir = args.out_dir or os.path.join(args.lvit_dir, "UNet_runs", args.task)
dataset = {"Covid19": "qata", "MosMedPlus": "mosmed"}[args.task]


class SegFolder(Dataset):
    """Same image/mask reading as LViT's ImageToImage2D, without the text."""

    def __init__(self, folder, transform):
        self.input_path = os.path.join(folder, "img")
        self.output_path = os.path.join(folder, "labelcol")
        self.mask_list = sorted(os.listdir(self.output_path))[:args.limit or None]
        self.transform = transform

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
        mask_filename = self.mask_list[idx]
        image_filename = mask_filename.replace("mask_", "")  # QaTa masks are "mask_<image name>"
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        image = cv2.resize(image, (args.img_size, args.img_size))
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (args.img_size, args.img_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1
        image, mask = correct_dims(image, mask)
        sample = self.transform({"image": image, "label": mask, "text": np.zeros((1, 1))})  # text unused
        return sample["image"], sample["label"], image_filename


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log(msg, path=None):
    print(msg, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def run_epoch(loader, model, criterion, optimizer, epoch, log_path):
    """One pass over loader (train if model.training), logged like LViT's train_one_epoch."""
    mode = "Train" if model.training else "Val"
    n_seen, time_sum, loss_sum, dice_sum, iou_sum = 0, 0.0, 0.0, 0.0, 0.0
    end = time.time()
    for i, (images, masks, _) in enumerate(loader, 1):
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = criterion(preds, masks.float())
        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        dice = float(criterion._show_dice(preds.detach().clone(), masks.float()))  # hard Dice at 0.5
        iou = iou_on_batch(masks, preds.detach().clone())
        batch_time = time.time() - end
        n = len(images)
        n_seen += n
        time_sum += n * batch_time
        loss_sum += n * loss.item()
        dice_sum += n * dice
        iou_sum += n * iou
        if i % args.print_frequency == 0 or i == len(loader):
            lr = " LR {:.2e}".format(optimizer.param_groups[0]["lr"]) if model.training else ""
            log("   [{}] Epoch: [{}][{}/{}]  Loss:{:.3f} (Avg {:.4f}) IoU:{:.3f} (Avg {:.4f}) "
                "Dice:{:.4f} (Avg {:.4f}){}   (AvgTime {:.2f})".format(
                    mode, epoch + 1, i, len(loader), loss.item(), loss_sum / n_seen, iou, iou_sum / n_seen,
                    dice, dice_sum / n_seen, lr, time_sum / n_seen), log_path)
        end = time.time()
    return loss_sum / n_seen, dice_sum / n_seen


def train():
    session = "Test_session_" + time.strftime("%m.%d_%Hh%M")
    session_dir = os.path.join(out_dir, session)
    os.makedirs(session_dir, exist_ok=True)
    log_path = os.path.join(session_dir, session + ".log")
    log("UNet on {} | {} | session {}".format(args.task, vars(args), session), log_path)

    train_tf = transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])])
    val_tf = ValGenerator(output_size=[args.img_size, args.img_size])
    train_set = SegFolder(os.path.join(data_dir, "Train_Folder"), train_tf)
    val_set = SegFolder(os.path.join(data_dir, "Val_Folder"), val_tf)
    log("train: {} images | val: {} images".format(len(train_set), len(val_set)), log_path)
    init = lambda w: random.seed(args.seed + w)  # same as train_model.worker_init_fn
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              worker_init_fn=init, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                            worker_init_fn=init, pin_memory=True)

    model = UNet(n_channels=3, n_classes=1).to(device)
    log("params: {}".format(sum(q.numel() for q in model.parameters())), log_path)
    write_run_config(session_dir, {
        "model": "U-Net", "dataset": dataset, "task": args.task, "data_dir": data_dir,
        "n_train": len(train_set), "n_val": len(val_set),
        "preprocessing": "image read as 3-channel uint8 and mask as grayscale (cv2), both resized to {0}x{0} "
                         "(bilinear), mask > 0 = lesion; train augmentation: LViT RandomGenerator (random rot90/flip "
                         "or rotation), val/test: LViT ValGenerator (to tensor, 0-1)".format(args.img_size),
        "prompts": "none (U-Net is image-only)",
        "training": {"loss": "WeightedDiceBCE(0.5, 0.5)", "optimizer": "Adam", "lr": args.lr,
                     "scheduler": "CosineAnnealingWarmRestarts(T_0=10, T_mult=1, eta_min=1e-4), stepped per epoch",
                     "batch_size": args.batch_size, "max_epochs": args.epochs,
                     "early_stopping": "stop after {} epochs without a new best val Dice".format(args.patience),
                     "checkpoint_rule": "best val Dice (hard, threshold 0.5), saved from epoch 6 on"},
        "seed": args.seed, "checkpoint": os.path.join(session_dir, "best_model-UNet.pth.tar"),
        "args": vars(args)})
    history_path = os.path.join(session_dir, "history.csv")
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_dice", "val_loss", "val_dice"])
    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)

    max_dice, best_epoch = 0.0, 1
    for epoch in range(args.epochs):
        log("\n========= Epoch [{}/{}] =========".format(epoch + 1, args.epochs), log_path)
        model.train()
        train_loss, train_dice = run_epoch(train_loader, model, criterion, optimizer, epoch, log_path)
        log("Validation", log_path)
        model.eval()
        with torch.no_grad():
            val_loss, val_dice = run_epoch(val_loader, model, criterion, optimizer, epoch, log_path)
        scheduler.step()
        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, train_dice, val_loss, val_dice])
        if val_dice > max_dice and epoch + 1 > 5:
            log("\t Saving best model, mean dice increased from: {:.4f} to {:.4f}".format(max_dice, val_dice),
                log_path)
            max_dice, best_epoch = val_dice, epoch + 1
            torch.save({"epoch": epoch, "model": "UNet", "state_dict": model.state_dict(), "val_loss": val_loss,
                        "val_dice": val_dice, "optimizer": optimizer.state_dict()},
                       os.path.join(session_dir, "best_model-UNet.pth.tar"))
        else:
            log("\t Mean dice:{:.4f} (best {:.4f} in epoch {})".format(val_dice, max_dice, best_epoch), log_path)
        early_stopping_count = epoch - best_epoch + 1  # same count as train_model.py
        log("\t early_stopping_count: {}/{}".format(early_stopping_count, args.patience), log_path)
        if early_stopping_count > args.patience:
            log("\t early_stopping!", log_path)
            break
    log("\nbest val Dice {:.4f} in epoch {}".format(max_dice, best_epoch), log_path)
    return session_dir


def predict_and_report(session_dir):
    """Best checkpoint -> probability maps for every val and test image, then the shared report (run_report.py)."""
    ckpt = torch.load(os.path.join(session_dir, "best_model-UNet.pth.tar"), map_location=device)
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log("UNet {} | session {} | best epoch {} (val Dice {:.4f})".format(
        args.task, os.path.basename(session_dir), ckpt["epoch"] + 1, ckpt["val_dice"]))
    for split, folder in [("val", "Val_Folder"), ("test", "Test_Folder")]:
        data = SegFolder(os.path.join(data_dir, folder), ValGenerator(output_size=[args.img_size] * 2))
        loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=args.workers)
        with torch.no_grad():
            for images, _, names in loader:
                prob = model(images.to(device))[0, 0].cpu().numpy()
                save_prob_mask(prob, os.path.join(session_dir, "preds_" + split, names[0]))
        log("saved {} {} predictions".format(len(data), split))
    finalize_run(dataset, session_dir, data_root=data_dir, require_all=not args.limit, best_epoch=ckpt["epoch"] + 1)


if __name__ == "__main__":
    seed_everything(args.seed)
    if args.test_only:
        predict_and_report(args.test_only if os.path.isabs(args.test_only) else os.path.join(out_dir, args.test_only))
    else:
        predict_and_report(train())
