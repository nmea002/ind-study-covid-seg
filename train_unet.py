"""Train + test the base U-Net (LViT/nets/UNet.py) on QaTa-COV19-v2 or MosMedData+, for comparison with LViT.

Everything except the model is kept the same as the LViT runs, using the LViT repo's own code where possible:
  - data: the same zips / folders (Train_Folder, Val_Folder, Test_Folder), so the same full split
  - augmentation: Load_Dataset.RandomGenerator / ValGenerator, images read and resized to 224 the same way
  - loss: utils.WeightedDiceBCE(0.5, 0.5); Adam lr 3e-4; cosine warm restarts (T_0=10, eta_min=1e-4) stepped per epoch
  - batch size 2, 20 epochs, seed 666; best model = best val Dice, only saved after epoch 5 (same rule as train_model.py)
  - test: per-image Dice and IoU at threshold 0.5, averaged over the test images (same as test_model.py)
The U-Net takes no text, so the BERT text embedding is skipped (bert_embedding is stubbed out to import Load_Dataset).

Usage (Colab, after unzipping a dataset zip to /content):
    python train_unet.py --task Covid19       # QaTa-COV19-v2
    python train_unet.py --task MosMedPlus    # MosMedData+
    python train_unet.py --task MosMedPlus --test_only <session folder>
"""
import argparse
import os
import random
import sys
import time
import types

import cv2
import numpy as np
import torch
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

p = argparse.ArgumentParser()
p.add_argument("--task", required=True, choices=["Covid19", "MosMedPlus"])
p.add_argument("--lvit_dir", default="/content/LViT", help="unzipped LViT folder (code + datasets/<task>)")
p.add_argument("--out_dir", default=None, help="where sessions are saved (default <lvit_dir>/UNet_runs/<task>)")
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--batch_size", type=int, default=2)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--img_size", type=int, default=224)
p.add_argument("--seed", type=int, default=666)
p.add_argument("--workers", type=int, default=2)
p.add_argument("--print_frequency", type=int, default=50)
p.add_argument("--limit", type=int, default=0, help="debug: only use the first N images of each folder")
p.add_argument("--test_only", default=None, help="session folder to test (skips training)")
args = p.parse_args()

# reuse the LViT repo's code; Load_Dataset imports bert_embedding at the top, which the U-Net doesn't need
sys.path.insert(0, args.lvit_dir)
sys.modules["bert_embedding"] = types.SimpleNamespace(BertEmbedding=None)
from Load_Dataset import RandomGenerator, ValGenerator, correct_dims  # noqa: E402
from nets.UNet import UNet  # noqa: E402
from utils import WeightedDiceBCE, iou_on_batch  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = os.path.join(args.lvit_dir, "datasets", args.task)
out_dir = args.out_dir or os.path.join(args.lvit_dir, "UNet_runs", args.task)


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
    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)

    max_dice, best_epoch = 0.0, 1
    for epoch in range(args.epochs):
        log("\n========= Epoch [{}/{}] =========".format(epoch + 1, args.epochs), log_path)
        model.train()
        run_epoch(train_loader, model, criterion, optimizer, epoch, log_path)
        log("Validation", log_path)
        model.eval()
        with torch.no_grad():
            val_loss, val_dice = run_epoch(val_loader, model, criterion, optimizer, epoch, log_path)
        scheduler.step()
        if val_dice > max_dice and epoch + 1 > 5:
            log("\t Saving best model, mean dice increased from: {:.4f} to {:.4f}".format(max_dice, val_dice),
                log_path)
            max_dice, best_epoch = val_dice, epoch + 1
            torch.save({"epoch": epoch, "model": "UNet", "state_dict": model.state_dict(), "val_loss": val_loss,
                        "val_dice": val_dice, "optimizer": optimizer.state_dict()},
                       os.path.join(session_dir, "best_model-UNet.pth.tar"))
        else:
            log("\t Mean dice:{:.4f} (best {:.4f} in epoch {})".format(val_dice, max_dice, best_epoch), log_path)
    log("\nbest val Dice {:.4f} in epoch {}".format(max_dice, best_epoch), log_path)
    return session_dir


def test(session_dir):
    log_path = os.path.join(session_dir, "test_results.txt")
    ckpt = torch.load(os.path.join(session_dir, "best_model-UNet.pth.tar"), map_location=device)
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    test_set = SegFolder(os.path.join(data_dir, "Test_Folder"), ValGenerator(output_size=[args.img_size] * 2))
    loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.workers)
    log("UNet test on {} | session {} | best epoch {} | {} test images".format(
        args.task, os.path.basename(session_dir), ckpt["epoch"] + 1, len(test_set)), log_path)
    dice_sum, iou_sum = 0.0, 0.0
    with torch.no_grad():
        for images, masks, _ in loader:
            pred = (model(images.to(device)) > 0.5).float()[0, 0].cpu().numpy()
            lab = masks[0].numpy().astype(np.float32)
            # same formulas as test_model.show_image_with_dice
            dice_sum += 2 * np.sum(lab * pred) / (np.sum(lab) + np.sum(pred) + 1e-5)
            iou_sum += jaccard_score(lab.reshape(-1), pred.reshape(-1), zero_division=0)
    log("dice_pred {:.4f}".format(dice_sum / len(test_set)), log_path)
    log("iou_pred {:.4f}".format(iou_sum / len(test_set)), log_path)


if __name__ == "__main__":
    seed_everything(args.seed)
    if args.test_only:
        test(args.test_only if os.path.isabs(args.test_only) else os.path.join(out_dir, args.test_only))
    else:
        test(train())
