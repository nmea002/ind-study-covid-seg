"""Freeze the train/val/test splits of QaTa-COV19-v2 and MosMedData+ so every model (U-Net, LViT, MedCLIPSeg,
MedSAM3) trains, validates and tests on exactly the same images.

QaTa-COV19-v2 (chest X-ray), raw data in LViT/datasets/archive/QaTa-COV19/QaTa-COV19-v2/:
    test        = the official Test Set (2113 images)
    train / val = the official Train Set (7145 images), split by the LViT authors' ID lists
                  (Train_ID_for_Covid19.xlsx: 5716, Val_ID_for_Covid19.xlsx: 1429)
    text        = the LViT authors' annotations (Train_Val_text.xlsx, Test_text.xlsx)

MosMedData+ (CT slices), raw data in LViT/datasets/MosMedData/{frames,masks}/ (2729 slices):
    no official split, so whole scans (filename prefix, e.g. "Jun_coronacases_case8_12.png" -> "Jun_coronacases_case8")
    go to one split only, so slices of the same CT scan never land in two splits:
    test  = ~20% of the slices, train / val = the rest, with ~12.5% of it held out as val
    text  = the LViT authors' annotations (their split is not used)
    Seeded, and reproduces the split used for the earlier MosMed runs (MosMed_train.csv / MosMed_test.csv).

Output (splits/):
    qata_split.csv    -- Image, Mask, text, split
    mosmed_split.csv  -- Image, Mask, text, scan, split
Image/Mask are filenames inside the raw image/mask folders above; split is train, val or test.
"""
import glob
import os
import random

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(ROOT, "LViT", "datasets")
OUT_DIR = os.path.join(ROOT, "splits")
SEED = 42

QATA_RAW = os.path.join(DATASETS, "archive", "QaTa-COV19", "QaTa-COV19-v2")
QATA_LVIT = os.path.join(DATASETS, "Covid19")
MOSMED = os.path.join(DATASETS, "MosMedData")
MOSMED_TEST_FRACTION = 0.2  # of all slices
MOSMED_VAL_FRACTION = 0.125  # of the non-test slices (~270, same size as the authors' val set)


def hold_out_scans(df, fraction, seed):
    """Shuffle the scans, then move whole scans out until the held-out part holds ~fraction of the slices."""
    scans = sorted(df["scan"].unique())
    random.Random(seed).shuffle(scans)
    sizes = df["scan"].value_counts()
    target = fraction * len(df)
    held, n = set(), 0
    for s in scans:
        if abs(n + sizes[s] - target) < abs(n - target):
            held.add(s)
            n += sizes[s]
    return df[~df["scan"].isin(held)], df[df["scan"].isin(held)]


def qata_split():
    train_ids = set(pd.read_excel(os.path.join(QATA_LVIT, "Train_ID_for_Covid19.xlsx"))["Image"])
    val_ids = set(pd.read_excel(os.path.join(QATA_LVIT, "Val_ID_for_Covid19.xlsx"))["Image"])
    text = pd.concat([pd.read_excel(os.path.join(QATA_LVIT, "Train_Folder", "Train_Val_text.xlsx")),
                      pd.read_excel(os.path.join(QATA_LVIT, "Test_Folder", "Test_text.xlsx"))])
    text = text.drop_duplicates("Image").set_index("Image")["Description"]

    rows = []
    for raw_set, splits in [("Train Set", None), ("Test Set", "test")]:
        masks = sorted(os.listdir(os.path.join(QATA_RAW, raw_set, "Ground-truths")))
        images = set(os.listdir(os.path.join(QATA_RAW, raw_set, "Images")))
        for m in masks:
            img = m[len("mask_"):]
            assert img in images, f"{raw_set}: no image for {m}"
            split = splits or ("train" if m in train_ids else "val" if m in val_ids else None)
            assert split, f"{m} is in neither the train nor the val ID list"
            rows.append({"Image": img, "Mask": m, "text": text.get(m), "split": split})
    df = pd.DataFrame(rows)
    assert df["text"].notna().all(), f"{df['text'].isna().sum()} QaTa images have no text annotation"
    assert (df["split"] == "train").sum() == len(train_ids) and (df["split"] == "val").sum() == len(val_ids)
    return df


def mosmed_split():
    text_csvs = glob.glob(os.path.join(MOSMED, "*_text_MosMedData*.csv"))
    text = pd.concat([pd.read_csv(f) for f in text_csvs]).drop_duplicates("Image").set_index("Image")["text"]

    images = sorted(os.listdir(os.path.join(MOSMED, "frames")))
    assert images == sorted(os.listdir(os.path.join(MOSMED, "masks"))), "frames and masks don't match"
    df = pd.DataFrame({"Image": images, "Mask": images})
    df["text"] = df["Image"].map(text)
    assert df["text"].notna().all(), f"{df['text'].isna().sum()} MosMed images have no text annotation"
    df["scan"] = df["Image"].str.replace(r"_\d+\.png$", "", regex=True)

    trainval, test = hold_out_scans(df, MOSMED_TEST_FRACTION, SEED)
    train, val = hold_out_scans(trainval, MOSMED_VAL_FRACTION, SEED)
    df["split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"

    # same split as the earlier runs
    for old_csv, splits in [("MosMed_train.csv", {"train", "val"}), ("MosMed_test.csv", {"test"})]:
        old_path = os.path.join(MOSMED, old_csv)
        if os.path.exists(old_path):
            assert set(pd.read_csv(old_path)["Image"]) == set(df.loc[df["split"].isin(splits), "Image"]), \
                f"split no longer matches {old_csv}"
    return df


def check_and_report(name, df, group=None):
    if group:  # no scan in two splits
        assert df.groupby(group)["split"].nunique().max() == 1, f"{name}: a {group} is in more than one split"
    print(name)
    for s in ["train", "val", "test"]:
        part = df[df["split"] == s]
        extra = f" / {part[group].nunique()} scans" if group else ""
        print(f"  {s:5s}: {len(part):5d} images{extra}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    qata = qata_split()
    check_and_report("QaTa-COV19-v2", qata)
    qata.to_csv(os.path.join(OUT_DIR, "qata_split.csv"), index=False)

    mosmed = mosmed_split()
    check_and_report("MosMedData+", mosmed, group="scan")
    mosmed.to_csv(os.path.join(OUT_DIR, "mosmed_split.csv"), index=False)
    print(f"wrote {OUT_DIR}")
