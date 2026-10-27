"""Package the LViT code + one dataset into a zip for the server (unpacked there by cluster/prepare_run.sh).

Uses the frozen split in splits/<dataset>_split.csv (run make_splits.py first), from the raw data:
    train -> Train_Folder, val -> Val_Folder (early stopping, threshold), test -> Test_Folder (only used at the end)

    python build_lvit_zip.py qata     # -> LViT_QaTa-COV19-v2.zip  (task Covid19)
    python build_lvit_zip.py mosmed   # -> LViT_MosMedData.zip     (task MosMedPlus)
    python build_lvit_zip.py all

Zip layout is what the LViT code (and train_unet.py) expects:
    LViT/<code files>
    LViT/datasets/<task>/{Train,Val,Test}_Folder/{img,labelcol}/<name>.png
    LViT/datasets/<task>/Train_Folder/Train_Val_text.xlsx   (Image = mask name, Description), train + val
    LViT/datasets/<task>/Test_Folder/Test_text.xlsx
    LViT/datasets/<task>/split_used.csv
The LViT code only knows the task names 'MoNuSeg' and 'Covid19', so for MosMed the zipped copies of
Config.py / train_model.py / test_model.py are patched to also accept 'MosMedPlus' (local files are untouched).
"""
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from evaluate import gt_path, image_path

ROOT = os.path.dirname(os.path.abspath(__file__))
LVIT = os.path.join(ROOT, "LViT")

CODE_FILES = ["Config.py", "Load_Dataset.py", "train_model.py", "test_model.py", "Train_one_epoch.py",
              "utils.py", "requirements.txt", "README.md", "LICENSE",
              "nets/LViT.py", "nets/pixlevel.py", "nets/textlevel.py", "nets/UNet.py", "nets/Vit.py"]

DATASETS = {
    "qata": {"task": "Covid19", "zip": "LViT_QaTa-COV19-v2.zip", "patches": []},
    "mosmed": {"task": "MosMedPlus", "zip": "LViT_MosMedData.zip", "patches": [
        # (file, old, new) -- each must match exactly once
        ("Config.py", "task_name = 'Covid19'", "task_name = 'MosMedPlus'"),
        ("train_model.py", "elif config.task_name == 'Covid19':", "elif config.task_name in ('Covid19', 'MosMedPlus'):"),
        ("test_model.py", 'elif config.task_name == "Covid19":', 'elif config.task_name in ("Covid19", "MosMedPlus"):'),
        ("test_model.py", "test_num = 2113", "test_num = len(os.listdir(config.test_dataset + 'img'))"),
        ("test_model.py", 'model_path = "./Covid19/" + model_type',
         'model_path = "./" + config.task_name + "/" + model_type'),
    ]},
}
FOLDERS = {"train": "Train_Folder", "val": "Val_Folder", "test": "Test_Folder"}


def png_bytes(path):
    """Grayscale PNGs saved as RGB (MosMed) are re-saved as 1-channel (~3x smaller zip; cv2.imread in
    Load_Dataset.py returns identical pixel values for both); anything else is copied unchanged."""
    img = Image.open(path)
    if img.mode != "RGB":
        with open(path, "rb") as f:
            return f.read()
    gray = img.convert("L")
    assert (np.array(img) == np.array(gray)[..., None]).all(), f"{path} is not grayscale"
    buf = io.BytesIO()
    gray.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def xlsx_bytes(df):
    # LViT looks the text up by mask filename
    buf = io.BytesIO()
    df.rename(columns={"Mask": "Image", "Image": "image_file", "text": "Description"})[["Image", "Description"]] \
        .to_excel(buf, index=False)
    return buf.getvalue()


def build(dataset):
    spec = DATASETS[dataset]
    split = pd.read_csv(os.path.join(ROOT, "splits", f"{dataset}_split.csv"))
    out_zip = os.path.join(ROOT, spec["zip"])
    base = "LViT/datasets/%s/" % spec["task"]

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in CODE_FILES:
            with open(os.path.join(LVIT, f), encoding="utf-8") as fh:
                src = fh.read()
            for pf, old, new in spec["patches"]:
                if pf == f:
                    assert src.count(old) == 1, f"patch target not found once in {f}: {old}"
                    src = src.replace(old, new)
            z.writestr("LViT/" + f, src)

        for _, r in split.iterrows():
            folder = base + FOLDERS[r["split"]]
            z.writestr(folder + "/img/" + r["Image"], png_bytes(image_path(dataset, r)), zipfile.ZIP_STORED)
            z.writestr(folder + "/labelcol/" + r["Mask"], png_bytes(gt_path(dataset, r)), zipfile.ZIP_STORED)
        # train_model.py reads one text file for both train and val (Covid19 code path)
        z.writestr(base + "Train_Folder/Train_Val_text.xlsx", xlsx_bytes(split[split["split"] != "test"]))
        z.writestr(base + "Test_Folder/Test_text.xlsx", xlsx_bytes(split[split["split"] == "test"]))
        z.writestr(base + "split_used.csv", split.to_csv(index=False))

    counts = split["split"].value_counts()
    print(f"{spec['zip']}: task {spec['task']} | train {counts['train']}, val {counts['val']}, test {counts['test']} "
          f"| {os.path.getsize(out_zip) / 1e6:.0f} MB")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which not in ("qata", "mosmed", "all"):
        sys.exit("usage: python build_lvit_zip.py <qata|mosmed|all>")
    for d in (["qata", "mosmed"] if which == "all" else [which]):
        build(d)
