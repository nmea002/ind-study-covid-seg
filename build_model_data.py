"""Export the frozen splits (splits/*.csv) in the input formats of MedCLIPSeg and MedSAM3, from the raw data.
Same images, same train/val/test as U-Net and LViT; only the layout and prompt wording differ.

    python build_model_data.py medclipseg qata      # or mosmed / all;  --zip also writes a zip per export
    python build_model_data.py medsam3 all --zip
    python build_model_data.py all all --zip

Masks are binarized the way evaluate.py and LViT read them (any value > 0 = lesion, saved as 0/255): MedCLIPSeg's
own loader treats values < 127 as background, which would drop the faint 12/17/21-valued lesion pixels in some
MosMed masks.

MedCLIPSeg -> model_data/medclipseg/<Covid19|MosMedPlus>/   (layout of MedCLIPSeg/assets/DATASETS.md)
    {Train,Val,Test}_Folder/{img,label}/<file>.png
    Prompts_Folder/Train_text.xlsx, Val_text.xlsx, Test_text_original.xlsx   (Image, Description, Ground Truth)
  Prompts: MedCLIPSeg's rewording of the LViT text ("Bilateral pulmonary infection, two infected areas, all left
  lung and lower right lung." -> "Bilateral pulmonary infection involving two regions with involvement of all left
  lung and the lower right lung"). For QaTa this reproduces their released prompts exactly (checked below when
  MedCLIPSeg/data/Covid19/Prompts_Folder is present); MosMed has no released prompts, so the same rule is applied.

MedSAM3 -> model_data/medsam3/<qata|mosmed>/   (COCO format read by MedSAM3/train_sam3_lora_native.py)
    {train,valid,test}/<file>.png + _annotations.coco.json
  One annotation per lesion = per 8-connected component of the mask (bbox + RLE segmentation), all with one
  category whose name is the text/concept prompt (CONCEPTS below). Images without a lesion get no annotations.
  Each image entry also keeps mask_file and the LViT text.
"""
import json
import os
import re
import shutil
import sys
import zipfile

import cv2
import numpy as np
import pandas as pd

from build_lvit_zip import png_bytes
from evaluate import gt_path, image_path, read_mask

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "model_data")
SPLITS = ["train", "val", "test"]

MEDCLIPSEG_NAMES = {"qata": "Covid19", "mosmed": "MosMedPlus"}
MEDCLIPSEG_FOLDERS = {"train": "Train_Folder", "val": "Val_Folder", "test": "Test_Folder"}
MEDCLIPSEG_TEXT = {"train": "Train_text.xlsx", "val": "Val_text.xlsx", "test": "Test_text_original.xlsx"}
MEDCLIPSEG_RELEASED = os.path.join(ROOT, "MedCLIPSeg", "data", "Covid19", "Prompts_Folder")

MEDSAM3_FOLDERS = {"train": "train", "val": "valid", "test": "test"}
# MedSAM3 is prompted with a short concept, not a sentence. Its concept vocabulary is not published yet, so check
# this phrasing (and alternatives) zero-shot on val before fine-tuning; changing it only needs a re-export.
CONCEPTS = {"qata": "covid-19 infection", "mosmed": "covid-19 infection"}

_LVIT_TEXT = re.compile(r"^(?P<side>\w+) pulmonary infection, (?P<n>\w+) infected areas, (?P<locs>.+?)\.?\s*$")


def medclipseg_prompt(text):
    """MedCLIPSeg's rewording of an LViT description (reverse-engineered from their released QaTa prompts)."""
    m = _LVIT_TEXT.match(text.strip())
    if not m:  # e.g. "one infected area" (singular) or "No pulmonary infection, zero infected areas."
        return text.strip().rstrip(".")
    locs = [loc.strip() for loc in m["locs"].split(" and ")]
    locs = [loc if loc.startswith("all") else "the " + loc for loc in locs]
    head = f"{m['side']} pulmonary infection involving {m['n']} regions"
    if len(locs) == 1:
        return f"{head} affecting {locs[0]}"
    return f"{head} with involvement of " + " and ".join(locs)


def binary_mask(dataset, row):
    return read_mask(gt_path(dataset, row)) > 0


def mask_png(mask):
    return cv2.imencode(".png", mask.astype(np.uint8) * 255)[1].tobytes()


def write_bytes(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def check_released_prompts(split):
    """QaTa: our prompts must equal MedCLIPSeg's released ones for every image."""
    if not os.path.isdir(MEDCLIPSEG_RELEASED):
        print("  (MedCLIPSeg's released QaTa prompts not found, skipping the check)")
        return
    released = {}
    for s, f in MEDCLIPSEG_TEXT.items():
        for _, r in pd.read_excel(os.path.join(MEDCLIPSEG_RELEASED, f)).iterrows():
            released[r["Ground Truth"]] = (s, r["Description"])
    ours = {m: (s, medclipseg_prompt(t)) for m, s, t in zip(split["Mask"], split["split"], split["text"])}
    assert ours == released, f"{sum(ours.get(k) != v for k, v in released.items())} QaTa prompts/splits differ"
    print("  prompts and splits identical to MedCLIPSeg's released QaTa prompts")


def export_medclipseg(dataset):
    split = pd.read_csv(os.path.join(ROOT, "splits", f"{dataset}_split.csv"))
    out = os.path.join(OUT, "medclipseg", MEDCLIPSEG_NAMES[dataset])
    shutil.rmtree(out, ignore_errors=True)
    for _, r in split.iterrows():
        folder = os.path.join(out, MEDCLIPSEG_FOLDERS[r["split"]])
        write_bytes(os.path.join(folder, "img", r["Image"]), png_bytes(image_path(dataset, r)))
        write_bytes(os.path.join(folder, "label", r["Mask"]), mask_png(binary_mask(dataset, r)))
    os.makedirs(os.path.join(out, "Prompts_Folder"))
    for s in SPLITS:
        part = split[split["split"] == s]
        pd.DataFrame({"Image": part["Image"], "Description": part["text"].map(medclipseg_prompt),
                      "Ground Truth": part["Mask"]}).to_excel(
            os.path.join(out, "Prompts_Folder", MEDCLIPSEG_TEXT[s]), index=False)
    if dataset == "qata":
        check_released_prompts(split)
    return out


def instances(mask):
    """One COCO annotation (bbox, RLE, area) per 8-connected lesion component."""
    from pycocotools import mask as mask_utils
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    anns = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        rle = mask_utils.encode(np.asfortranarray(labels == i, dtype=np.uint8))
        rle["counts"] = rle["counts"].decode("ascii")
        anns.append({"bbox": [x, y, w, h], "area": area, "segmentation": rle, "iscrowd": 0})
    return anns


def export_medsam3(dataset):
    from pycocotools import mask as mask_utils
    split = pd.read_csv(os.path.join(ROOT, "splits", f"{dataset}_split.csv"))
    out = os.path.join(OUT, "medsam3", dataset)
    shutil.rmtree(out, ignore_errors=True)
    category = {"id": 1, "name": CONCEPTS[dataset], "supercategory": "lesion"}
    for s in SPLITS:
        folder = os.path.join(out, MEDSAM3_FOLDERS[s])
        images, annotations = [], []
        for image_id, (_, r) in enumerate(split[split["split"] == s].iterrows(), 1):
            mask = binary_mask(dataset, r)
            write_bytes(os.path.join(folder, r["Image"]), png_bytes(image_path(dataset, r)))
            images.append({"id": image_id, "file_name": r["Image"], "height": mask.shape[0], "width": mask.shape[1],
                           "mask_file": r["Mask"], "text": r["text"]})
            anns = instances(mask)
            if anns:  # the instances must add up to the binary mask exactly
                union = np.any([mask_utils.decode(a["segmentation"]) for a in anns], axis=0)
                assert (union == mask).all(), f"instances do not reproduce the mask of {r['Image']}"
            for a in anns:
                annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": 1, **a})
        with open(os.path.join(folder, "_annotations.coco.json"), "w") as f:
            json.dump({"info": {"description": f"{dataset} {s} split from splits/{dataset}_split.csv"},
                       "images": images, "annotations": annotations, "categories": [category]}, f)
        n_empty = len(images) - len({a["image_id"] for a in annotations})
        print(f"  {MEDSAM3_FOLDERS[s]:5s}: {len(images)} images, {len(annotations)} lesion instances, "
              f"{n_empty} without lesion")
    return out


def zip_folder(folder):
    out_zip = folder.rstrip("/\\") + ".zip"
    base = os.path.dirname(folder)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as z:
        for d, _, files in os.walk(folder):
            for f in files:
                z.write(os.path.join(d, f), os.path.relpath(os.path.join(d, f), base))
    print(f"  wrote {out_zip} ({os.path.getsize(out_zip) / 1e6:.0f} MB)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--zip"]
    if len(args) != 2 or args[0] not in ("medclipseg", "medsam3", "all") or args[1] not in ("qata", "mosmed", "all"):
        sys.exit("usage: python build_model_data.py <medclipseg|medsam3|all> <qata|mosmed|all> [--zip]")
    exporters = {"medclipseg": export_medclipseg, "medsam3": export_medsam3}
    for model in (exporters if args[0] == "all" else [args[0]]):
        for dataset in (["qata", "mosmed"] if args[1] == "all" else [args[1]]):
            print(f"{model} / {dataset}")
            folder = exporters[model](dataset)
            counts = pd.read_csv(os.path.join(ROOT, "splits", f"{dataset}_split.csv"))["split"].value_counts()
            print(f"  -> {folder}  (train {counts['train']}, val {counts['val']}, test {counts['test']})")
            if "--zip" in sys.argv:
                zip_folder(folder)
