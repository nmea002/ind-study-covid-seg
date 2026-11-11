"""Packs the code the demo needs into medseg_demo_code.zip (project root), for running the Colab notebook without
pushing to GitHub: upload the zip to the same Drive folder as the checkpoint and the notebook uses it.

    python demo/make_code_bundle.py

No weights, data or images go in (the checkpoint is uploaded separately).
"""
import os
import zipfile

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDERS = ["demo", "medsam3_setup", "medclipseg_setup", "splits"]
FILES = ["evaluate.py", "run_report.py", "DEMO_README.md"]
SKIP_DIRS = {"weights", "__pycache__", "screenshots", ".ipynb_checkpoints"}
SKIP_EXT = (".pth", ".pt", ".png", ".jpg", ".jpeg", ".zip", ".pyc")

out = os.path.join(PROJECT, "medseg_demo_code.zip")
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for folder in FOLDERS:
        for root, dirs, files in os.walk(os.path.join(PROJECT, folder)):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                if not f.endswith(SKIP_EXT):
                    path = os.path.join(root, f)
                    z.write(path, os.path.relpath(path, PROJECT).replace(os.sep, "/"))
                    n += 1
    for f in FILES:
        z.write(os.path.join(PROJECT, f), f)
        n += 1
print(f"{n} files -> {out} ({os.path.getsize(out) / 1e3:.0f} kB)")
