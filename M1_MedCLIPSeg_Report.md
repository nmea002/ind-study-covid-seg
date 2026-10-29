# M1: MedCLIPSeg Fine-Tuning

> **Status:** MosMedData+ is finished. QaTa-COV19-v2 is still training: at the time of writing it had run 37 epochs, with a best val Dice of 0.8194 at epoch 25. Its rows say *pending* until the run finishes.

## Setup
| | |
|---|---|
| Code | Official repository [HealthX-Lab/MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg). Our wrappers are `medclipseg_setup/train.py` and `medclipseg_setup/predict.py`, and the model code is unchanged. |
| Pretrained parts | UniMedCLIP ViT-B/16 image encoder (`unimed_clip_vit_b16.pt`, from TahaKoleilat/MedCLIPSeg) and the BiomedBERT text encoder (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`). |
| Configs | `medclipseg_setup/configs/{Covid19,MosMedPlus}.yaml` use the authors' QaTa settings for both datasets. The only changes are the data paths and using the best val checkpoint instead of the last one. |
| Data | Exported by `build_model_data.py` from the frozen splits (`splits/*_split.csv`) into MedCLIPSeg's folder layout (`img/`, `label/`, `Prompts_Folder/*.xlsx`). |
| Hardware | One NVIDIA GB10 (DGX Spark, ARM64), shared with other runs. Slurm jobs are limited to 12 h. |
| Seed | 42 |

### Splits (same for every model)
| Dataset | Train | Val | Test | Notes |
|---|---|---|---|---|
| QaTa-COV19-v2 (chest X-ray) | 5,716 | 1,429 | 2,113 | official test set; train/val from the LViT ID lists |
| MosMedData+ (chest CT slices) | 1,853 | 327 | 549 | split by scan (40 / 6 / 25 scans), seed 42, so no patient appears in two splits |

### Preprocessing
- **Images:** read as 3-channel, resized to 224×224, normalized with ImageNet mean and std (the CLIP preprocessing).
- **Masks:** resized nearest-neighbour; any value above 0 counts as lesion.
- **Training augmentation:** random rotation of ±20° with probability 0.5, MedCLIPSeg's default.

### Text prompts
Each image gets its own prompt: the LViT text annotation, reworded the way MedCLIPSeg's released prompts are. For QaTa the rewording reproduces their released prompts exactly. We apply the same rule to MosMedData+. Example:

| LViT annotation | Prompt used |
|---|---|
| Bilateral pulmonary infection, two infected areas, middle left lung and middle right lung. | Bilateral pulmonary infection involving two regions with involvement of the middle left lung and the middle right lung |

### Training
| | |
|---|---|
| Loss | 0.5 Dice + 0.5 BCE + 0.1 CLIP loss |
| Optimizer | Adam, lr 3e-4, CosineAnnealingLR (T_max 50, eta_min 1e-4) |
| Batch size | 24 |
| Epochs | at most 50; early stopping after 20 epochs with no new best val Dice (we did not use the earlier fixed 20-epoch cap) |
| Checkpoint | the epoch with the best val Dice (threshold 0.5, one forward pass) |
| Inference | MedCLIPSeg's test protocol: the mean probability over 30 stochastic forward passes |
| Threshold | swept from 0.05 to 0.95 on **val**; the best val Dice value is then used once on test |

### Evaluation
`evaluate.py` scores every model the same way, at the original ground-truth resolution (QaTa 224×224, MosMed 512×512). Predicted probability maps are resized bilinearly to that resolution before thresholding.
- **Metrics:** Dice and foreground IoU, plus HD95 and ASSD in pixels. HD95 and ASSD match medpy's definitions.
- **Empty masks:** if the prediction and ground truth are both empty, the image scores perfectly. If only one is empty, Dice and IoU are 0, and HD95 and ASSD are set to the image diagonal.

### Sanity check
Before reporting, we re-predicted the val set from the best MosMed checkpoint in a separate run of `predict.py`. It scored val Dice 0.662 at threshold 0.5, consistent with the training log. This check matters because an early version of `predict.py` dropped the wrong dimension of the model output and produced empty masks. That bug was fixed before any results here were made.

## Results
Test set, mean ± std per image. Each checkpoint and threshold was chosen on val only.

| Dataset | Split | Dice | IoU | HD95 (px) | ASSD (px) | Threshold | Best epoch | Stopped |
|---|---|---|---|---|---|---|---|---|
| QaTa-COV19-v2 | val | *pending* | | | | | | |
| QaTa-COV19-v2 | test | *pending* | | | | | | |
| MosMedData+ | val | 0.6729 ± 0.2110 | 0.5378 ± 0.1975 | 61.35 ± 114.75 | 28.79 ± 105.78 | 0.40 | 45 | at the 50-epoch cap |
| MosMedData+ | test | 0.5043 ± 0.2921 | 0.3848 ± 0.2472 | 152.64 ± 245.03 | 120.64 ± 251.06 | 0.40 | 45 | at the 50-epoch cap |

### Comparison with the baselines on the same splits (test Dice)
| Model | QaTa-COV19-v2 | MosMedData+ |
|---|---|---|
| U-Net | 0.7818 | 0.5540 |
| LViT | **0.8191** | 0.5579 |
| MedCLIPSeg | *pending* | 0.5043 |
| MedSAM3, zero-shot (see M2) | 0.7677 | **0.6180** |

### Observations
- **MosMedData+, val vs test:** val Dice is 0.67 but test Dice only 0.50. Val has 6 scans and test has 25, so the checkpoint and threshold were chosen on few patients and did not carry over well.
  - The large HD95 and ASSD standard deviations come from slices where only one of the prediction and the ground truth is empty. Each of those gets the image-diagonal penalty of about 724 px.
  - Per-image scores are in `per_image_test.csv`.
- **MosMedData+, training length:** val Dice was still improving slowly at epoch 50, with the best at epoch 45. With more epochs allowed, the result might change.

## Figures and files
In `runs/medclipseg/<Covid19|MosMedPlus>/trained_models/seed42/` (training curves, examples, per-image scores, settings, metrics and checkpoints):

| File | Contents |
|---|---|
| `curves.png` | training and val loss and Dice per epoch, with the selected epoch marked |
| `examples/` | 5 test images, each shown as image \| ground truth \| prediction \| overlay; every model uses the same 5 images |
| `per_image_val.csv`, `per_image_test.csv` | metrics for each image |
| `run_config.json` | preprocessing, prompts, training settings, checkpoint and seed |
| `metrics.json` | the val threshold sweep, plus val and test summaries |
| `*_best_dice.pth` | the checkpoint (on the cluster only) |

### MosMedData+ examples (test)
![](runs/medclipseg/MosMedPlus/trained_models/seed42/examples/Jun_coronacases_case4_72.png)
![](runs/medclipseg/MosMedPlus/trained_models/seed42/examples/Jun_coronacases_case7_121.png)
![](runs/medclipseg/MosMedPlus/trained_models/seed42/examples/Morozov_study_0282_15.png)

### MosMedData+ training curves
![](runs/medclipseg/MosMedPlus/trained_models/seed42/curves.png)

### QaTa-COV19-v2 examples and curves
*Pending: added when the run finishes.*

## Reproduce
```bash
cd cluster
sbatch -J mcs-mosmed train_medclipseg.slurm mosmed        # trains (resuming), then predicts and writes the report
BUDGET_HOURS=9 sbatch -J final train_pack.slurm medclipseg:qata:final ...
```
