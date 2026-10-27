"""Package LViT + MosMedData+ into LViT_MosMedData.zip for training on the cluster (cluster/prepare_run.sh).

Uses the frozen split in splits/mosmed_split.csv (run make_splits.py first):
    train -> Train_Folder, val -> Val_Folder (early stopping), test -> Test_Folder (only used once at the end)

Zip layout matches what the LViT code expects (same as the QaTa zip):
    LViT/datasets/MosMedPlus/{Train,Val,Test}_Folder/{img,labelcol}/<name>.png
    LViT/datasets/MosMedPlus/Train_Folder/Train_Val_text.xlsx   (Image, Description)
    LViT/datasets/MosMedPlus/Test_Folder/Test_text.xlsx
The LViT code only knows the task names 'MoNuSeg' and 'Covid19', so the zipped copies of
Config.py / train_model.py / test_model.py are patched to also accept 'MosMedPlus' (local files are untouched).
"""
import io
import os
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
LVIT = os.path.join(ROOT, "LViT")
DATA_DIR = os.path.join(LVIT, "datasets", "MosMedData")
OUT_ZIP = os.path.join(ROOT, "LViT_MosMedData.zip")
SPLIT_CSV = os.path.join(ROOT, "splits", "mosmed_split.csv")
TASK = "MosMedPlus"

CODE_FILES = ["Config.py", "Load_Dataset.py", "train_model.py", "test_model.py", "Train_one_epoch.py",
              "utils.py", "requirements.txt", "README.md", "LICENSE",
              "nets/LViT.py", "nets/pixlevel.py", "nets/textlevel.py", "nets/UNet.py", "nets/Vit.py"]

# (file, old, new) -- each must match exactly once
PATCHES = [
    ("Config.py", "task_name = 'Covid19'", "task_name = '%s'" % TASK),
    ("train_model.py", "elif config.task_name == 'Covid19':", "elif config.task_name in ('Covid19', '%s'):" % TASK),
    ("test_model.py", 'elif config.task_name == "Covid19":', 'elif config.task_name in ("Covid19", "%s"):' % TASK),
    ("test_model.py", "test_num = 2113", "test_num = len(os.listdir(config.test_dataset + 'img'))"),
    ("test_model.py", 'model_path = "./Covid19/" + model_type',
     'model_path = "./" + config.task_name + "/" + model_type'),
]


def gray_png_bytes(path):
    """The PNGs are grayscale saved as RGB; re-save as 1-channel (~3x smaller zip).
    cv2.imread in Load_Dataset.py returns identical pixel values for both."""
    img = Image.open(path)
    gray = img.convert("L")
    if img.mode == "RGB":
        assert (np.array(img) == np.array(gray)[..., None]).all(), f"{path} is not grayscale"
    buf = io.BytesIO()
    gray.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def xlsx_bytes(df):
    buf = io.BytesIO()
    df.rename(columns={"text": "Description"})[["Image", "Description"]].to_excel(buf, index=False)
    return buf.getvalue()


split = pd.read_csv(SPLIT_CSV)
train, val, test = (split[split["split"] == s] for s in ["train", "val", "test"])
assert not set(train["scan"]) & set(val["scan"]) and not set(train["scan"]) & set(test["scan"])     and not set(val["scan"]) & set(test["scan"])

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for f in CODE_FILES:
        with open(os.path.join(LVIT, f), encoding="utf-8") as fh:
            src = fh.read()
        for pf, old, new in PATCHES:
            if pf == f:
                assert src.count(old) == 1, f"patch target not found once in {f}: {old}"
                src = src.replace(old, new)
        z.writestr("LViT/" + f, src)

    base = "LViT/datasets/%s/" % TASK
    for folder, df in [("Train_Folder", train), ("Val_Folder", val), ("Test_Folder", test)]:
        for name in df["Image"]:
            z.writestr(base + folder + "/img/" + name, gray_png_bytes(os.path.join(DATA_DIR, "frames", name)),
                       zipfile.ZIP_STORED)
            z.writestr(base + folder + "/labelcol/" + name, gray_png_bytes(os.path.join(DATA_DIR, "masks", name)),
                       zipfile.ZIP_STORED)
    # train_model.py reads one text file for both train and val (Covid19 code path)
    z.writestr(base + "Train_Folder/Train_Val_text.xlsx", xlsx_bytes(pd.concat([train, val])))
    z.writestr(base + "Test_Folder/Test_text.xlsx", xlsx_bytes(test))
    z.writestr(base + "split_used.csv", split.to_csv(index=False))

print(f"train: {len(train)} slices / {train['scan'].nunique()} scans")
print(f"val:   {len(val)} slices / {val['scan'].nunique()} scans: {sorted(val['scan'].unique())}")
print(f"test:  {len(test)} slices / {test['scan'].nunique()} scans")
print(f"wrote {OUT_ZIP} ({os.path.getsize(OUT_ZIP) / 1e6:.0f} MB)")
