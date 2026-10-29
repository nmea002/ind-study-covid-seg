# M2: MedSAM3 LoRA Fine-Tuning

> **Status:** zero-shot inference with the released MedSAM3 LoRA is finished on both datasets. LoRA fine-tuning on both datasets is running in the current cluster job. Its rows say *pending* until the job finishes.

## Setup
| | |
|---|---|
| Code | [Joey-S-Liu/MedSAM3](https://github.com/Joey-S-Liu/MedSAM3). We wrote three wrappers: `medsam3_setup/common.py` (model loading and the rule that merges instances), `predict.py` (zero-shot) and `train.py` (fine-tuning). The model, LoRA layers, losses and training step are MedSAM3's. |
| Base model | SAM3, `facebook/sam3` (a gated Hugging Face model) |
| LoRA weights | released MedSAM3 v1 LoRA (`lal-Joey/MedSAM3_v1`), rank 16, alpha 32, dropout 0.1, applied to every component: vision and text encoders, geometry encoder, DETR encoder and decoder, mask decoder |
| Check on load | the LoRA layers are applied to 458 modules, and all 916 LoRA tensors are checked to load with matching names (`load_lora_checked`) |
| Config | `medsam3_setup/configs/{qata,mosmed}.yaml`, taken from MedSAM3's `full_lora_config.yaml`. Only the data paths, the batch size and the epoch cap are changed. |
| Data | the frozen splits, converted to COCO format by `build_model_data.py`: one annotation per connected lesion region, with polygon and box. Images without a lesion have no annotations. |
| Hardware | one NVIDIA GB10 (DGX Spark, ARM64), shared with other runs; 12 h Slurm jobs |
| Seed | 42 |

The splits are the same as for every other model:
- **QaTa-COV19-v2:** 5,716 / 1,429 / 2,113 images (train / val / test)
- **MosMedData+:** 1,853 / 327 / 549 slices, split by scan

### Preprocessing
- **Images:** read as RGB, resized to 1008×1008, normalized to [-1, 1] (SAM3's transforms).
- **Masks:** any value above 0 counts as lesion.

### Prompts: text only
- **The prompt:** one fixed concept prompt per dataset, **"covid-19 infection"**, the same for every image.
- **What the model never sees:** no boxes, points or ground-truth masks at inference, on val or test.

### Turning instances into one binary mask
SAM3 returns several detections, each with a score and a mask. This rule is the same for every split, run and dataset:
1. Keep detections with a score above 0.5.
2. Apply NMS at IoU 0.5 on their boxes.
3. Pixel probability = the maximum mask probability over the kept detections, or 0 if none are kept.
4. Threshold the probability map. The threshold is swept from 0.05 to 0.95 on **val**, and the best val Dice value is used once on test.

### Fine-tuning (in progress)
| | |
|---|---|
| Start | the released MedSAM3 v1 LoRA (epoch 0) |
| Trainable parameters | LoRA only; SAM3 stays frozen |
| Loss | SAM3's losses as in `train_sam3_lora_native.py`: Hungarian matching; L1 + GIoU on boxes; IABCE on class and presence; focal + Dice on masks |
| Optimizer | AdamW, lr 5e-5, weight decay 0.01. The optimizer steps every batch, with no gradient accumulation and no scheduler, the same as MedSAM3's native training script. |
| Batch size | 2 |
| Epochs | at most 50; early stopping after 20 epochs with no new best val Dice |
| Checkpoint | best val Dice; after every epoch, val is predicted with the text prompt as above and scored at threshold 0.5 |
| Time budget | training stops early when its time budget runs out, even in the middle of an epoch, so that predicting and the report still fit in the job. When that happens, `stop_reason.txt` records it and the report says so. |

### Evaluation
`evaluate.py` scores every model the same way, at the original ground-truth resolution (QaTa 224×224, MosMed 512×512). Predicted probability maps are resized bilinearly to that resolution before thresholding.
- **Metrics:** Dice and foreground IoU, plus HD95 and ASSD in pixels. HD95 and ASSD match medpy's definitions.
- **Empty masks:** if the prediction and ground truth are both empty, the image scores perfectly. If only one is empty, Dice and IoU are 0, and HD95 and ASSD are set to the image diagonal.

## Results
Mean ± std per image. The threshold was chosen on val only.

### Zero-shot (released MedSAM3 v1 LoRA, no training on our data)
| Dataset | Split | Dice | IoU | HD95 (px) | ASSD (px) | Threshold |
|---|---|---|---|---|---|---|
| QaTa-COV19-v2 | val | 0.7510 ± 0.2247 | 0.6454 ± 0.2497 | 35.32 ± 44.81 | 12.01 ± 29.49 | 0.50 |
| QaTa-COV19-v2 | test | 0.7677 ± 0.2254 | 0.6691 ± 0.2559 | 32.67 ± 40.26 | 10.07 ± 20.62 | 0.50 |
| MosMedData+ | val | 0.7386 ± 0.2207 | 0.6232 ± 0.2199 | 61.15 ± 134.84 | 34.37 ± 130.16 | 0.35 |
| MosMedData+ | test | 0.6180 ± 0.2744 | 0.4967 ± 0.2538 | 93.28 ± 196.23 | 69.37 ± 195.10 | 0.35 |

### LoRA fine-tuned
| Dataset | Split | Dice | IoU | HD95 (px) | ASSD (px) | Threshold | Best epoch | Stopped |
|---|---|---|---|---|---|---|---|---|
| QaTa-COV19-v2 | val | *pending* | | | | | | |
| QaTa-COV19-v2 | test | *pending* | | | | | | |
| MosMedData+ | val | *pending* | | | | | | |
| MosMedData+ | test | *pending* | | | | | | |

### Comparison with the other models on the same splits (test Dice)
| Model | QaTa-COV19-v2 | MosMedData+ |
|---|---|---|
| U-Net | 0.7818 | 0.5540 |
| LViT | **0.8191** | 0.5579 |
| MedCLIPSeg (M1) | *pending* | 0.5043 |
| MedSAM3, zero-shot | 0.7677 | **0.6180** |
| MedSAM3, LoRA fine-tuned | *pending* | *pending* |

### Observations
- **Without training on our data:** MedSAM3 zero-shot gets the best Dice, IoU and HD95 on MosMedData+ of all models so far. On QaTa-COV19-v2 it is within 5 Dice points of LViT.
- **Threshold choice:** the zero-shot val Dice changes by less than 0.01 across thresholds 0.3–0.6 on both datasets. The probability maps are close to binary, so the result does not depend much on the threshold.
- **MosMedData+, val vs test:** Dice drops from 0.74 on val to 0.62 on test. MedCLIPSeg drops the same way, which points to differences between the 6 val scans and the 25 test scans rather than to the model.
- **Speed:** with the GPU shared by eight processes, one fine-tuning step took about 18.5 s. That is why fine-tuning runs as a separate job with a time budget.

## Model selection (val only)
Per the plan, MedCLIPSeg and MedSAM3 are compared on the **same val cases**, mainly by Dice. The model is chosen from these val numbers, not from the test results.

| Dataset | MedCLIPSeg val Dice | MedSAM3 zero-shot val Dice | MedSAM3 fine-tuned val Dice | Selected |
|---|---|---|---|---|
| QaTa-COV19-v2 (X-ray) | *pending* (0.8194 in the training log) | 0.7510 | *pending* | *pending* |
| MosMedData+ (CT) | 0.6729 | **0.7386** | *pending* | *pending* |

Inference cost (time per image) is the secondary criterion and has not been measured yet.
- **MedCLIPSeg:** runs 30 stochastic forward passes at 224 px.
- **MedSAM3:** runs one forward pass at 1008 px with a much larger model.

## Figures and files
In `runs/medsam3/<qata|mosmed>/<zeroshot|finetune>/` (examples, per-image scores, settings, metrics and LoRA weights):

| File | Contents |
|---|---|
| `examples/` | 5 test images, each shown as image \| ground truth \| prediction \| overlay; every model uses the same 5 images |
| `per_image_val.csv`, `per_image_test.csv` | metrics for each image |
| `run_config.json` | preprocessing, prompt, instance rule, training settings, checkpoint and seed |
| `metrics.json` | the val threshold sweep, plus val and test summaries |
| `history.csv`, `curves.png`, `stop_reason.txt` | fine-tuning only |
| `best_lora_weights.pt` | fine-tuned LoRA (on the cluster only) |

### QaTa-COV19-v2, zero-shot examples (test)
![](runs/medsam3/qata/zeroshot/examples/sub-S09475_ses-E24600_run-1_bp-chest_cr.png)
![](runs/medsam3/qata/zeroshot/examples/sub-S09678_ses-E18547_run-1_bp-chest_vp-ap_dx.png)
![](runs/medsam3/qata/zeroshot/examples/sub-S09827_ses-E25861_run-1_bp-chest_vp-ap_dx.png)

### MosMedData+, zero-shot examples (test)
![](runs/medsam3/mosmed/zeroshot/examples/Jun_coronacases_case4_72.png)
![](runs/medsam3/mosmed/zeroshot/examples/Jun_coronacases_case7_121.png)
![](runs/medsam3/mosmed/zeroshot/examples/Morozov_study_0282_15.png)

### Fine-tuned examples and training curves
*Pending: added when the runs finish.*

## Reproduce
```bash
cd cluster
bash setup_medsam3.sh                                      # login node, once (after `hf auth login`)
sbatch -J ms3-zs-qata train_medsam3.slurm qata zeroshot    # zero-shot: predict val + test, report
BUDGET_HOURS=9 sbatch -J ms3-qata train_medsam3.slurm qata # LoRA fine-tuning, then predict + report
```
