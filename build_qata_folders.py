"""Fill LViT/datasets/Covid19/{Train,Val,Test}_Folder with the full QaTa-COV19-v2 split from make_splits.py.

LViT (and train_unet.py) read QaTa from LViT/datasets/Covid19/<split>_Folder/{img,labelcol}/, not from the raw download.
This copies each image/mask from the raw archive into the folder of its split (splits/qata_split.csv) and removes
any file that does not belong there, so every folder holds exactly its split. Files already in place are left alone,
and the text spreadsheets (Train_Val_text.xlsx, Test_text.xlsx) are not touched.

Run make_splits.py first.
"""
import os
import shutil

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "LViT", "datasets", "archive", "QaTa-COV19", "QaTa-COV19-v2")
OUT = os.path.join(ROOT, "LViT", "datasets", "Covid19")
SPLIT_CSV = os.path.join(ROOT, "splits", "qata_split.csv")

FOLDERS = {"train": "Train_Folder", "val": "Val_Folder", "test": "Test_Folder"}
RAW_SET = {"train": "Train Set", "val": "Train Set", "test": "Test Set"}

split = pd.read_csv(SPLIT_CSV)
for s, folder in FOLDERS.items():
    part = split[split["split"] == s]
    for sub, col, raw_sub in [("img", "Image", "Images"), ("labelcol", "Mask", "Ground-truths")]:
        dst_dir = os.path.join(OUT, folder, sub)
        os.makedirs(dst_dir, exist_ok=True)
        wanted = set(part[col])
        present = set(os.listdir(dst_dir))
        for name in sorted(present - wanted):
            os.remove(os.path.join(dst_dir, name))
        for name in sorted(wanted - present):
            shutil.copy2(os.path.join(RAW, RAW_SET[s], raw_sub, name), os.path.join(dst_dir, name))
        assert set(os.listdir(dst_dir)) == wanted
        print(f"{folder}/{sub}: {len(wanted)} files ({len(wanted - present)} copied, {len(present - wanted)} removed)")
