"""Choose a modality's fixed default prompt on the VAL split (never test), for the demo's "image only" case.

Every candidate prompt is used as the single prompt for all val images of that modality's dataset, with the demo's
checkpoint and val threshold, and scored with the shared evaluate.py. Candidates are the K most frequent train
descriptions (the wording the model was trained on), plus anything passed with --extra.

    python demo/select_default_prompt.py --modality "Chest X-ray" --top_k 5
    python demo/select_default_prompt.py --modality "Chest CT slice" --extra "covid-19 infection"   # MedSAM3

Put the winner into demo_config.yaml as default_prompt. Results are written to demo/default_prompt_<dataset>.json.
"""
import argparse
import json
import os
import sys
import tempfile

import pandas as pd
import yaml
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from backends import PROJECT, load_backend  # noqa: E402

sys.path.insert(1, PROJECT)
from evaluate import evaluate, summarize  # noqa: E402
from run_report import save_prob_mask  # noqa: E402

EXPORT = {"qata": "Covid19", "mosmed": "MosMedPlus"}  # model_data/medclipseg/<export>: images + GT of the split

p = argparse.ArgumentParser()
p.add_argument("--modality", required=True)
p.add_argument("--config", default=os.path.join(HERE, "demo_config.yaml"))
p.add_argument("--dataset", choices=["qata", "mosmed"], help="default: X-ray -> qata, CT -> mosmed")
p.add_argument("--top_k", type=int, default=5)
p.add_argument("--extra", nargs="*", default=[])
p.add_argument("--limit", type=int, default=None, help="debug: first N val images only")
a = p.parse_args()

with open(a.config) as f:
    spec = yaml.safe_load(f)["modalities"][a.modality]
dataset = a.dataset or ("qata" if "x-ray" in a.modality.lower() else "mosmed")
root = os.path.join(PROJECT, "model_data", "medclipseg", EXPORT[dataset])

train_text = pd.read_excel(os.path.join(root, "Prompts_Folder", "Train_text.xlsx"))
candidates = list(train_text["Description"].value_counts().index[:a.top_k]) if a.top_k else []
candidates += [c for c in [spec["default_prompt"], *a.extra] if c not in candidates]

val_dir = os.path.join(root, "Val_Folder", "img")
images = sorted(os.listdir(val_dir))[:a.limit]
backend = load_backend(spec)
results = []
for prompt in candidates:
    with tempfile.TemporaryDirectory() as pred_dir:
        for name in tqdm(images, desc=prompt[:50], mininterval=30):
            save_prob_mask(backend.predict(os.path.join(val_dir, name), prompt), os.path.join(pred_dir, name))
        s = summarize(evaluate(dataset, "val", pred_dir, spec["threshold"], data_root=root,
                               require_all=a.limit is None))
    results.append({"prompt": prompt, "val_dice": s["dice"]["mean"], "val_iou": s["iou"]["mean"], "n": s["n_images"]})
    print(f"val Dice {s['dice']['mean']:.4f} | {prompt}", flush=True)

results.sort(key=lambda r: -r["val_dice"])
out = os.path.join(HERE, f"default_prompt_{dataset}.json")
with open(out, "w") as f:
    json.dump({"modality": a.modality, "checkpoint": spec["checkpoint"], "threshold": spec["threshold"],
               "split": "val", "results": results}, f, indent=2)
print(f"\nbest default prompt (val): {results[0]['prompt']!r} -> {out}")
