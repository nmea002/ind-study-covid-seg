# Data and Splits

This project uses two datasets:

- **QaTa-COV19-v2**: chest X-rays with masks of the COVID-19 infected regions.
- **MosMedData+**: CT slices with masks of the infected regions.

Every model (U-Net, LViT, MedCLIPSeg, MedSAM3) uses the same frozen train/val/test split, listed in `splits/`. Do not move images between splits.

## The data

| Dataset | Raw data | Contents |
|---|---|---|
| QaTa-COV19-v2 | `LViT/datasets/archive/QaTa-COV19/QaTa-COV19-v2/` | Official Train Set: 7145 images and masks. Official Test Set: 2113 images and masks. |
| MosMedData+ | `LViT/datasets/MosMedData/frames/` and `masks/` | 2729 CT slices and masks from 71 CT scans. |

Text annotations come from the LViT authors. Each image has one description, such as *"Bilateral pulmonary infection, two infected areas, all left lung and all right lung."*

- QaTa: `LViT/datasets/Covid19/Train_Folder/Train_Val_text.xlsx` and `Test_Folder/Test_text.xlsx`
- MosMed: `LViT/datasets/MosMedData/*_text_MosMedData+*.csv`

In QaTa, the mask for `covid_1.png` is named `mask_covid_1.png`. In MosMed, an image and its mask have the same filename.

`LViT/datasets/archive/QaTa-COV19/` also contains QaTa-COV19-v1, Early-QaTa-COV19 and the control groups. This project does not use them.

## How the folders relate

The LViT code (and `train_unet.py`, which reuses LViT's data loader) does not read the raw folders. It loads data from a fixed layout named after its task:

```
LViT/datasets/<task_name>/Train_Folder/img/                  images
LViT/datasets/<task_name>/Train_Folder/labelcol/             masks
LViT/datasets/<task_name>/Train_Folder/Train_Val_text.xlsx   text for train + val
LViT/datasets/<task_name>/Val_Folder/{img,labelcol}/
LViT/datasets/<task_name>/Test_Folder/{img,labelcol}/  +  Test_text.xlsx
```

Each dataset therefore has a raw copy and an LViT-layout copy:

| Folder | What it is | Built by |
|---|---|---|
| `LViT/datasets/archive/QaTa-COV19/QaTa-COV19-v2/` | Raw QaTa download | (downloaded) |
| `LViT/datasets/Covid19/` | QaTa in the LViT layout. The LViT code calls QaTa `Covid19`. | `build_qata_folders.py` |
| `LViT/datasets/MosMedData/` | Raw MosMed, plus the LViT authors' text CSVs | (downloaded) |
| `LViT/datasets/MosMedPlus/` | MosMed in the LViT layout. Exists only inside `LViT_MosMedData.zip`. | `build_lvit_zip.py` |

`LViT/datasets/Covid19/` is not a third dataset. It contains the same files as the raw QaTa download, byte for byte, sorted into train/val/test folders. It also holds the files that define the QaTa train/val split and its text: `Train_ID_for_Covid19.xlsx`, `Val_ID_for_Covid19.xlsx` and the text spreadsheets. **Do not delete it.**

The other folders in `LViT/datasets/` (`MoNuSeg`, `Dataset_BUSI_with_GT_Clean`) came with the LViT repo and are not used.

## The splits

| Dataset | Train | Val | Test |
|---|---|---|---|
| QaTa-COV19-v2 | 5716 | 1429 | 2113 |
| MosMedData+ | 1853 slices / 40 scans | 327 slices / 6 scans | 549 slices / 25 scans |

The splits are saved as:

- `splits/qata_split.csv`: columns `Image, Mask, text, split`
- `splits/mosmed_split.csv`: columns `Image, Mask, text, scan, split`

`split` is `train`, `val` or `test`. `Image` and `Mask` are filenames inside the raw folders.

### QaTa-COV19-v2

- **Test:** the official Test Set.
- **Train / val:** the official Train Set, divided using the LViT authors' ID lists (`Train_ID_for_Covid19.xlsx`, `Val_ID_for_Covid19.xlsx`).

Because this is the standard split, QaTa results can be compared directly with the LViT paper.

### MosMedData+

MosMed has no official split. The LViT authors' CSVs do define one, but it **leaks across CT scans**:

| Splits compared | Scans in both |
|---|---|
| Train and test | 4 |
| Val and test | 4 |
| Train and val | 2 |

Neighbouring slices of the same scan are nearly identical. With that split, the model would be tested partly on near-copies of its training images, and the scores would come out too high.

Our split is made per scan instead. A slice's scan is its filename without the slice number: `Jun_coronacases_case8_12.png` belongs to scan `Jun_coronacases_case8`. Every slice of a scan goes to the same split:

1. Whole scans are shuffled (seed 42) and moved into **test** until test holds about 20% of the slices.
2. From the remaining scans, whole scans are moved into **val** until val holds about 12.5% of the remainder.
3. Everything else is **train**.

This reproduces the split used in the earlier MosMed runs (`LViT/datasets/MosMedData/MosMed_train.csv` and `MosMed_test.csv`), and `make_splits.py` checks that it still does. Because the split differs from the LViT paper's, MosMed results are not directly comparable with the paper. Mention this when reporting.

### Rules for every model

- Train on the full train split and choose checkpoints and thresholds by **val Dice only**.
- Evaluate on test only after all settings are fixed.
- Use the text in the split CSVs as each image's text prompt, for models that take text.

## Scripts

Run from the project root, in this order:

| Script | What it does |
|---|---|
| `python make_splits.py` | Builds `splits/qata_split.csv` and `splits/mosmed_split.csv` from the raw data and checks them. Re-running produces the same split. |
| `python build_qata_folders.py` | Fills `LViT/datasets/Covid19/{Train,Val,Test}_Folder` with exactly the QaTa split, copying from the raw archive. Running it again changes nothing. |
| `python build_lvit_zip.py all` | Builds `LViT_QaTa-COV19-v2.zip` and `LViT_MosMedData.zip` (or pass `qata` / `mosmed` for one): the LViT code plus the full split of a dataset in the LViT layout, for the server. |
| `python build_model_data.py all all --zip` | Exports the same split for MedCLIPSeg (`model_data/medclipseg/<Covid19/MosMedPlus>/`, its folder layout + prompt spreadsheets) and MedSAM3 (`model_data/medsam3/<qata/mosmed>/`, COCO format with one instance per lesion), plus a zip of each for the server. Masks are binarized (> 0 = lesion). |
| `python evaluate.py --dataset <qata/mosmed> --split <val/test> --pred_dir <dir>` | Shared evaluation for all models: Dice, IoU, HD95 and ASSD on a folder of predicted mask PNGs, measured at the ground-truth resolution. |
| `python run_report.py --dataset <qata/mosmed> --run_dir <dir>` | Finishes a run the same way for every model: picks the threshold on val, evaluates val and test once, plots training curves and saves example figures. Training scripts call it automatically. |

## On the school server

The server gets its data from the zips. Copy the whole project folder (the code, `splits/` and both zips; the raw data under `LViT/datasets/` is not needed), then from `cluster/`:

```
bash prepare_run.sh qata   ../LViT_QaTa-COV19-v2.zip     # once per dataset: unzip + patch into <project>/runs/<name>/
bash prepare_run.sh mosmed ../LViT_MosMedData.zip
sbatch -J lvit-qata   train_lvit.slurm qata              # LViT
sbatch -J lvit-mosmed train_lvit.slurm mosmed
sbatch -J unet-qata   train_unet.slurm qata              # U-Net, same run folder and data
sbatch -J unet-mosmed train_unet.slurm mosmed
```

Each run ends with the shared report (`run_report.py`) in its session folder.

MedCLIPSeg (Milestone 1): our training and prediction wrappers and configs are in `medclipseg_setup/`; the MedCLIPSeg repo itself is used unchanged. Also copy `medclipseg_setup/` and `model_data/medclipseg/*.zip` to the server, then from `cluster/`:

```
bash setup_medclipseg.sh                                     # once, on the login node: clone, env, weights, unzip data
sbatch -J mcs-sanity train_medclipseg.slurm qata sanity      # released QaTa checkpoint on our val split
sbatch -J mcs-qata   train_medclipseg.slurm qata             # train + predict + report (resubmits itself near 12 h)
sbatch -J mcs-mosmed train_medclipseg.slurm mosmed
```

Output: `runs/medclipseg/<Covid19|MosMedPlus>/trained_models/seed42/`.
