"""After LViT training: best checkpoint -> probability maps for every val and test image, then the shared report.

Run from inside a prepared run's LViT folder (train_lvit.slurm does this), so Config.py / nets / Load_Dataset are
the run's own copies:
    python predict_lvit.py --project_dir <project folder with evaluate.py, run_report.py, splits/> [--session <name>]

Writes into the session folder (<task>/LViT/<session>/):
    preds_val/, preds_test/   probability maps (0-255 = 0-1), named like the images
    run_config.json           preprocessing, prompts, training settings, checkpoint, seed
and then run_report.finalize_run adds metrics.json, per_image_*.csv, curves.png (from history.csv) and examples/.
Text is fed exactly as in training: the image's description, padded by utils.read_text, BERT-embedded,
first 10 tokens (Train_one_epoch.py).
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())  # the run's LViT code

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import Config as config  # noqa: E402
from Load_Dataset import ImageToImage2D, ValGenerator  # noqa: E402
from nets.LViT import LViT  # noqa: E402
from utils import read_text  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--project_dir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
p.add_argument("--session", default=None, help="session folder name (default: the newest)")
args = p.parse_args()
sys.path.insert(1, args.project_dir)
from run_report import finalize_run, save_prob_mask, write_run_config  # noqa: E402

DATASETS = {"Covid19": "qata", "MosMedPlus": "mosmed"}
dataset = DATASETS[config.task_name]
model_type = config.model_name
sessions_dir = os.path.join(config.task_name, model_type)
session = args.session or sorted(os.listdir(sessions_dir))[-1]
session_dir = os.path.join(sessions_dir, session)
ckpt_path = os.path.join(session_dir, "models", "best_model-" + model_type + ".pth.tar")
data_root = os.path.abspath(os.path.join("datasets", config.task_name))

ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
state = {k.replace("module.", "", 1): v for k, v in ckpt["state_dict"].items()}  # if saved from DataParallel
model = LViT(config.get_CTranS_config(), n_channels=config.n_channels, n_classes=config.n_labels)
model.load_state_dict(state)
model = model.cuda().eval()
best_epoch = ckpt["epoch"] + 1
print(f"LViT {config.task_name} | session {session} | best epoch {best_epoch}", flush=True)

write_run_config(session_dir, {
    "model": "LViT", "dataset": dataset, "task": config.task_name, "data_dir": data_root,
    "preprocessing": "image read as 3-channel uint8 and mask as grayscale (cv2), both resized to {0}x{0} (bilinear), "
                     "mask > 0 = lesion; train augmentation: LViT RandomGenerator (random rot90/flip or rotation), "
                     "val/test: LViT ValGenerator (to tensor, 0-1)".format(config.img_size),
    "prompts": "the LViT authors' text annotation of each image (text column of the split CSV), padded to >= 9 words "
               "with ' EOF XXX' (utils.read_text), embedded with BERT-base uncased (last layer, per word, "
               "cluster/bert_embedding.py), first 10 tokens fed to the model",
    "training": {"loss": "WeightedDiceBCE(0.5, 0.5)", "optimizer": "Adam", "lr": config.learning_rate,
                 "scheduler": "CosineAnnealingWarmRestarts(T_0=10, T_mult=1, eta_min=1e-4)" if config.cosineLR else None,
                 "batch_size": config.batch_size, "max_epochs": config.epochs,
                 "early_stopping": "stop after {} epochs without a new best val Dice".format(
                     config.early_stopping_patience),
                 "checkpoint_rule": "best val Dice (hard, threshold 0.5), saved from epoch 6 on",
                 "transformer": {"num_heads": config.get_CTranS_config().transformer.num_heads,
                                 "num_layers": config.get_CTranS_config().transformer.num_layers}},
    "seed": config.seed, "checkpoint": os.path.abspath(ckpt_path), "best_epoch": best_epoch})

for split, folder, text_file in [("val", config.val_dataset, config.task_dataset + "Train_Val_text.xlsx"),
                                 ("test", config.test_dataset, config.test_dataset + "Test_text.xlsx")]:
    data = ImageToImage2D(folder, config.task_name, read_text(text_file), ValGenerator(output_size=[config.img_size] * 2),
                          image_size=config.img_size)
    loader = DataLoader(data, batch_size=1, shuffle=False)
    with torch.no_grad():
        for sample, names in loader:
            text = sample["text"]
            if text.shape[1] > 10:  # same as Train_one_epoch.py
                text = text[:, :10, :]
            prob = model(sample["image"].cuda(), text.cuda())[0, 0].cpu().numpy()
            save_prob_mask(prob, os.path.join(session_dir, "preds_" + split, names[0].replace("mask_", "")))
    print(f"saved {len(data)} {split} predictions", flush=True)

finalize_run(dataset, session_dir, data_root=data_root, best_epoch=best_epoch)
