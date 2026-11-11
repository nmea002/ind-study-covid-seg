# Demo: text-prompted COVID-19 lesion segmentation

A Gradio app that segments COVID-19 lesions in **2D chest X-rays and chest CT slices**, using one validated checkpoint per modality. It runs inference only, with no training.

> Research demo only. It is not a medical device and must not be used for clinical diagnosis.

## Selected models

The plan's rule is to compare MedCLIPSeg and MedSAM3 on the **same val cases**, choosing mainly by Dice, and to fix the choice before final test reporting. Different models won on the two modalities, so the demo loads one checkpoint per modality.

| Modality | Dataset | MedCLIPSeg val Dice | MedSAM3 zero-shot val Dice | MedSAM3 fine-tuned val Dice | Demo uses |
|---|---|---|---|---|---|
| Chest X-ray | QaTa-COV19-v2 | **0.8229** | 0.7510 | 0.7510 (epoch 0) | MedCLIPSeg, epoch 48 checkpoint, threshold 0.50 |
| Chest CT slice | MosMedData+ | 0.6729 | **0.7386** | 0.7386 (epoch 0) | MedSAM3, released v1 LoRA, threshold 0.35 |

MedSAM3 fine-tuning stopped on its time budget without beating the released LoRA on val (best epoch 0), so the released LoRA is the MedSAM3 checkpoint. For reference, the test Dice of the selected checkpoints, computed after selection, is 0.8414 (X-ray) and 0.6180 (CT). The selection lives in `demo/demo_config.yaml`.

## What the app does

| Input | Behaviour |
|---|---|
| Image only | Runs with the modality's fixed default prompt and shows which prompt was used. X-ray: *"Bilateral pulmonary infection involving two regions with involvement of all left lung and all right lung"*, the most frequent QaTa training description. CT: *"covid-19 infection"*, the concept MedSAM3 was evaluated with. |
| Image + text | Runs with the user's prompt. |
| Text only | Shows *"Please upload an image before running segmentation."* No inference is run. |

Both models are prompt-conditioned. "Image only" means the interface supplies a default prompt, not that the model ignores text.

The interface has:
- **Inputs:** a `.png` / `.jpg` / `.jpeg` upload, a modality choice (Chest X-ray / Chest CT slice), an optional prompt box and a **Segment** button.
- **Outputs:** the input image, the binary mask and an overlay, with the mask and overlay downloadable as PNG.
- **Run details:** the model, checkpoint, prompt, threshold and inference time. No Dice is shown, because uploads have no ground truth.
- **Errors:** unsupported formats, unreadable files, missing checkpoints and failed inference each produce a readable message instead of a crash.

The preprocessing and prediction rule match the evaluated runs:
- **MedCLIPSeg:** 224 px input, mean of 30 stochastic passes.
- **MedSAM3:** 1008 px input, detections with score > 0.5, NMS at IoU 0.5, then the max over instance masks.
- **Both:** the probability map is resized to the uploaded image size and thresholded with `>` at the val-selected threshold, the same rule `evaluate.py` uses.

## Run it on Google Colab

Open `demo/MedSeg_Demo_Colab.ipynb` in Colab (browser or the VS Code Colab extension), choose a **GPU** runtime and run the cells in order. You need:
1. **SAM3 access:** a Hugging Face token from an account with access to the gated `facebook/sam3` model.
2. **The trained MedCLIPSeg checkpoint:** in the Drive folder set as `DRIVE_DIR` in the notebook (default `MyDrive/Colab Schoolwork/CPS 4881/medseg_demo_weights/`), named `medclipseg_qata_best_dice.pth`. Copy it off the cluster from `runs/medclipseg/Covid19/trained_models/seed42/MedCLIPSeg_unimedclip_ViT-B-16_best_dice.pth`.
3. **The code**, either from GitHub (cloned automatically) or as `medseg_demo_code.zip` in the same Drive folder. Make the zip with `python demo/make_code_bundle.py`. If the zip is there, the notebook uses it instead of GitHub, which lets you test unpushed changes.

The released MedSAM3 LoRA is downloaded automatically. The last cell prints a public `gradio.live` link.

## Run it elsewhere (cluster GPU node, workstation)

```bash
hf auth login                      # access to facebook/sam3
bash demo/setup_demo_env.sh        # in a fresh env: clones both repos, installs deps, downloads SAM3 + MedSAM3 LoRA
cp <MedCLIPSeg best checkpoint> demo/weights/medclipseg_qata_best_dice.pth
python demo/app.py --share --preload
```

`python demo/app.py --mock` starts the interface with a fake model, for UI work without a GPU. Its output is labelled *MOCK*.

## Files

| File | Purpose |
|---|---|
| `demo/app.py` | Gradio interface and input handling |
| `demo/backends.py` | MedCLIPSeg / MedSAM3 inference wrappers (reuses `medsam3_setup/common.py`) |
| `demo/demo_config.yaml` | modality → model, checkpoint, val threshold, default prompt |
| `demo/select_default_prompt.py` | picks a modality's default prompt by val Dice (never test) |
| `demo/setup_demo_env.sh` | one-environment install for both models |
| `demo/MedSeg_Demo_Colab.ipynb` | runnable Colab notebook |

## Screenshots

*To add after the first GPU run: image only, image + text, and text only.*

## Limitations

- **2D only:** single 2D images, one slice at a time.
- **Prompts:** must stay within the trained concepts, which are COVID-19 pulmonary infection and lesion location in the lungs. The models were not trained to segment other findings, so other prompts give meaningless output.
- **CT generalisation:** MosMedData+ results vary between scans (val Dice 0.74 vs test 0.62 for MedSAM3 zero-shot), so expect uneven quality on new CT data.
- **Default X-ray prompt:** it names both whole lungs, which may bias the mask toward bilateral disease. `select_default_prompt.py` can re-choose it on val.
