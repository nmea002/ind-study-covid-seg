"""Make a fixed train/test split of MosMedData+ for fair model comparisons.

Splits by scan (filename prefix, e.g. "Jun_coronacases_case8_12.png" -> "Jun_coronacases_case8"),
so slices from the same CT scan never end up in both train and test.
The text annotation for each image is taken from the LViT authors' CSVs (their split is not used).

Output (in LViT/datasets/MosMedData/):
    MosMed_train.csv, MosMed_test.csv  -- columns: Image, text, scan
"""
import glob
import os
import random

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LViT", "datasets", "MosMedData")
TEST_FRACTION = 0.2
SEED = 42

# text annotations from the authors' Train/Val/Test CSVs, merged into one lookup
text_csvs = [f for f in glob.glob(os.path.join(DATA_DIR, "*_text_MosMedData*.csv"))]
text = pd.concat([pd.read_csv(f) for f in text_csvs]).drop_duplicates("Image").set_index("Image")["text"]

images = sorted(os.listdir(os.path.join(DATA_DIR, "frames")))
assert images == sorted(os.listdir(os.path.join(DATA_DIR, "masks"))), "frames and masks don't match"
missing = [f for f in images if f not in text.index]
assert not missing, f"{len(missing)} images have no text annotation, e.g. {missing[:5]}"

df = pd.DataFrame({"Image": images})
df["text"] = df["Image"].map(text)
df["scan"] = df["Image"].str.replace(r"_\d+\.png$", "", regex=True)

# shuffle the scans with a fixed seed, then move scans into test until it holds ~20% of the slices
scans = sorted(df["scan"].unique())
random.Random(SEED).shuffle(scans)
sizes = df["scan"].value_counts()
target = TEST_FRACTION * len(df)
test_scans, n_test = set(), 0
for s in scans:
    if abs(n_test + sizes[s] - target) < abs(n_test - target):
        test_scans.add(s)
        n_test += sizes[s]

test = df[df["scan"].isin(test_scans)]
train = df[~df["scan"].isin(test_scans)]
assert not set(train["scan"]) & set(test["scan"])

train.to_csv(os.path.join(DATA_DIR, "MosMed_train.csv"), index=False)
test.to_csv(os.path.join(DATA_DIR, "MosMed_test.csv"), index=False)
print(f"train: {len(train)} slices from {train['scan'].nunique()} scans")
print(f"test:  {len(test)} slices from {test['scan'].nunique()} scans ({len(test) / len(df):.1%})")
print("test scans:", sorted(test_scans))
