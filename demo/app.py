"""Gradio demo: text-prompted COVID-19 lesion segmentation on 2D chest X-rays and CT slices. Inference only.

    python demo/app.py                    # local URL
    python demo/app.py --share            # public *.gradio.live link (Colab, cluster)
    python demo/app.py --preload          # load both checkpoints at start instead of on first use
    python demo/app.py --mock             # UI development without a GPU or weights (fake masks, clearly labelled)

Which model and checkpoint serve each modality is set in demo/demo_config.yaml.
"""
import argparse
import os
import sys
import tempfile
import threading
import time

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backends import PROJECT, binarize, load_backend  # noqa: E402

ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg")
NO_IMAGE_ERROR = "Please upload an image before running segmentation."
DISCLAIMER = ("Research demo for 2D chest X-ray and CT-slice images. It is **not** a medical device and must not "
              "be used for clinical diagnosis. No accuracy is shown for uploads, since there is no ground-truth "
              "mask to compare with.")


class MockBackend:
    """Fake model for working on the interface without a GPU: marks the brightest smooth regions."""
    name = "MOCK (not a real model)"

    def __init__(self, spec, device=None):
        self.checkpoint, self.epoch = "none (mock)", None

    def predict(self, image_path, prompt):
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("the file could not be decoded as an image")
        blur = cv2.GaussianBlur(gray.astype(np.float32) / 255, (0, 0), 8)
        return np.clip((blur - blur.mean()) / (blur.std() + 1e-6) / 4 + 0.5, 0, 1)


class Demo:
    def __init__(self, config_path, mock=False):
        with open(config_path) as f:
            self.modalities = yaml.safe_load(f)["modalities"]
        self.mock = mock
        self.backends = {}
        self.lock = threading.Lock()
        self.out_dir = tempfile.mkdtemp(prefix="medseg_demo_")

    def backend(self, modality):
        with self.lock:  # one load at a time; requests for the same modality wait for it
            if modality not in self.backends:
                spec = self.modalities[modality]
                self.backends[modality] = MockBackend(spec) if self.mock else load_backend(spec)
            return self.backends[modality]

    def default_prompt_hint(self, modality):
        spec = self.modalities[modality]
        return f"Leave empty to use the default prompt for {modality}: *\"{spec['default_prompt']}\"*"

    def segment(self, image_file, modality, prompt):
        """Returns (input, mask, overlay, mask file, overlay file, info markdown). Never raises."""
        empty = (None, None, None, None, None)
        prompt = (prompt or "").strip()

        if image_file is None:  # covers "text only, no image": nothing is run
            return (*empty, f"### Error\n{NO_IMAGE_ERROR}")
        path = image_file if isinstance(image_file, str) else getattr(image_file, "name", None)
        if not path or not path.lower().endswith(ALLOWED_EXTENSIONS):
            return (*empty, "### Error\nUnsupported file format. Please upload a **.png**, **.jpg** or **.jpeg** "
                            "image.")
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            return (*empty, "### Error\nThe file could not be read as an image. It may be corrupted; please try "
                            "another .png, .jpg or .jpeg file.")
        if modality not in self.modalities:
            return (*empty, "### Error\nPlease choose a modality: Chest X-ray or Chest CT slice.")

        spec = self.modalities[modality]
        used_default = not prompt
        prompt = prompt or spec["default_prompt"]
        try:
            backend = self.backend(modality)
            start = time.time()
            prob = backend.predict(path, prompt)
            seconds = time.time() - start
        except Exception as e:  # noqa: BLE001 - shown to the user instead of a stack trace
            return (*empty, f"### Inference failed\n`{type(e).__name__}: {e}`\n\nCheck that the checkpoint in "
                            "`demo/demo_config.yaml` exists and that a GPU runtime is selected.")

        mask = binarize(prob, spec["threshold"])
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay = make_overlay(rgb, mask)
        stem = os.path.splitext(os.path.basename(path))[0]
        req_dir = tempfile.mkdtemp(dir=self.out_dir)
        mask_path = os.path.join(req_dir, f"{stem}_mask.png")
        overlay_path = os.path.join(req_dir, f"{stem}_overlay.png")
        cv2.imwrite(mask_path, mask * 255)
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        lesion = 100 * mask.mean()
        info = "\n".join([
            "### Result",
            f"- **Modality:** {modality}",
            f"- **Model:** {backend.name}",
            f"- **Checkpoint:** `{short_path(backend.checkpoint)}`"
            + (f" (epoch {backend.epoch})" if backend.epoch else ""),
            f"- **Prompt used:** \"{prompt}\" " + ("(automatic default for this modality)" if used_default
                                                   else "(your prompt)"),
            f"- **Threshold:** {spec['threshold']} (chosen on the validation set)",
            f"- **Predicted lesion area:** {lesion:.1f}% of the image"
            + (" (no lesion found at this threshold)" if not mask.any() else ""),
            f"- **Inference time:** {seconds:.1f} s",
        ])
        return rgb, mask * 255, overlay, mask_path, overlay_path, info


def short_path(path):
    """Checkpoint path relative to the project when it is inside it (no local directory names in the UI)."""
    try:
        rel = os.path.relpath(path, PROJECT)
    except ValueError:  # different drive on Windows
        return os.path.basename(path)
    return os.path.basename(path) if rel.startswith("..") else rel.replace(os.sep, "/")


def make_overlay(rgb, mask, color=(255, 40, 40), alpha=0.45):
    out = rgb.astype(np.float32)
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, np.float32)
    out = out.astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, color, max(1, round(max(rgb.shape[:2]) / 300)))
    return out


def build_ui(demo):
    import gradio as gr

    modalities = list(demo.modalities)
    with gr.Blocks(title="COVID-19 Lesion Segmentation Demo") as ui:
        gr.Markdown("# COVID-19 lesion segmentation (text-prompted)\n" + DISCLAIMER)
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.File(label="Image (.png, .jpg, .jpeg)", file_types=list(ALLOWED_EXTENSIONS),
                                   type="filepath")
                modality = gr.Radio(modalities, value=modalities[0], label="Modality")
                prompt = gr.Textbox(label="Text prompt (optional)", lines=2,
                                    placeholder="e.g. Unilateral pulmonary infection, one infected area, "
                                                "lower right lung")
                hint = gr.Markdown(demo.default_prompt_hint(modalities[0]))
                run = gr.Button("Segment", variant="primary")
                info = gr.Markdown()
            with gr.Column(scale=2):
                with gr.Row():
                    shown_in = gr.Image(label="Input image", interactive=False)
                    shown_mask = gr.Image(label="Predicted mask", interactive=False, image_mode="L")
                    shown_overlay = gr.Image(label="Overlay", interactive=False)
                with gr.Row():
                    mask_file = gr.File(label="Download mask (PNG)")
                    overlay_file = gr.File(label="Download overlay (PNG)")

        modality.change(demo.default_prompt_hint, modality, hint)
        run.click(demo.segment, [image_in, modality, prompt],
                  [shown_in, shown_mask, shown_overlay, mask_file, overlay_file, info])
    return ui


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(PROJECT, "demo", "demo_config.yaml"))
    p.add_argument("--share", action="store_true", help="create a public gradio.live link")
    p.add_argument("--preload", action="store_true", help="load every modality's checkpoint before serving")
    p.add_argument("--mock", action="store_true", help="fake model, for UI work without a GPU")
    p.add_argument("--port", type=int, default=None)
    a = p.parse_args()
    demo = Demo(a.config, mock=a.mock)
    if a.preload:
        for m in demo.modalities:
            print(f"loading {m} ...", flush=True)
            demo.backend(m)
    build_ui(demo).queue(default_concurrency_limit=1).launch(share=a.share, server_port=a.port)
