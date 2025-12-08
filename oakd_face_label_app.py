#!/usr/bin/env python
"""
Simple OAK-D capture + labeling + ID demo.

Features:
- Capture single snapshots from OAK-D RGB camera to a local folder.
- Label stored images (e.g., select a few photos and assign "John Doe").
- Identify a new photo by comparing its FaceNet embedding to labeled embeddings.

Notes:
- Runs embeddings on host CPU/GPU using facenet-pytorch; no HIP kernels on OAK-D.
- Captures use the OAK-D onboard color camera; neural nets can be added to the pipeline later.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import cv2
import gradio as gr
import numpy as np
import torch
import torchvision.transforms as T
from facenet_pytorch import InceptionResnetV1

import depthai as dai


DEFAULT_CAPTURE_DIR = Path("./oakd_captures")
LABELS_JSON = DEFAULT_CAPTURE_DIR / "labels.json"


def ensure_dirs(capture_dir: Path):
    capture_dir.mkdir(parents=True, exist_ok=True)
    if not LABELS_JSON.exists():
        LABELS_JSON.write_text("{}", encoding="utf-8")


def load_labels() -> Dict[str, str]:
    if LABELS_JSON.exists():
        return json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    return {}


def save_labels(labels: Dict[str, str]) -> None:
    LABELS_JSON.write_text(json.dumps(labels, indent=2), encoding="utf-8")


def build_facenet(device: torch.device) -> InceptionResnetV1:
    model = InceptionResnetV1(pretrained="vggface2", classify=False)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def preprocess_image(img_bgr: np.ndarray) -> torch.Tensor:
    # Convert BGR -> RGB, resize to 160x160 (FaceNet default), normalize to [0,1]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = T.ToPILImage()(img_rgb)
    tfm = T.Compose([
        T.Resize((160, 160)),
        T.ToTensor(),  # [0,1], C,H,W
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # match FaceNet
    ])
    return tfm(pil)  # torch.float32


@torch.no_grad()
def embed_image(model: InceptionResnetV1, device: torch.device, img_bgr: np.ndarray) -> torch.Tensor:
    x = preprocess_image(img_bgr).unsqueeze(0).to(device)
    emb = model(x)  # [1,512]
    return emb.squeeze(0).cpu()


def capture_from_oakd(capture_dir: Path) -> str:
    pipeline = dai.Pipeline()
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(160, 160)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    xout = pipeline.createXLinkOut()
    xout.setStreamName("preview")
    cam.preview.link(xout.input)

    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name="preview", maxSize=1, blocking=True)
        frame = q.get().getCvFrame()  # BGR numpy

    filename = f"capture_{len(list(capture_dir.glob('capture_*.png'))):05d}.png"
    out_path = capture_dir / filename
    cv2.imwrite(out_path.as_posix(), frame)
    return out_path.as_posix()


def list_images(capture_dir: Path) -> List[str]:
    return sorted([p.as_posix() for p in capture_dir.glob("*.png")])


def label_images(selected: List[str], label: str):
    labels = load_labels()
    for img in selected:
        name = Path(img).name
        labels[name] = label
    save_labels(labels)
    return f"Labeled {len(selected)} images as '{label}'."


def identify_image(
    image_path: str,
    device: torch.device,
    facenet: InceptionResnetV1,
):
    if not image_path:
        return "No image provided."

    labels = load_labels()
    if not labels:
        return "No labeled gallery. Label images first."

    # Build gallery embeddings
    gallery_embs = {}
    for img_name, lbl in labels.items():
        img_path = DEFAULT_CAPTURE_DIR / img_name
        if not img_path.exists():
            continue
        img = cv2.imread(img_path.as_posix())
        if img is None:
            continue
        emb = embed_image(facenet, device, img)
        gallery_embs.setdefault(lbl, []).append(emb)

    if not gallery_embs:
        return "No valid labeled images found."

    # Embed query
    img_q = cv2.imread(image_path)
    if img_q is None:
        return "Failed to read query image."
    emb_q = embed_image(facenet, device, img_q)

    # Compare with mean embedding per label
    best_label = None
    best_score = -1.0
    for lbl, embs in gallery_embs.items():
        stack = torch.stack(embs, dim=0)
        mean_emb = stack.mean(dim=0)
        # cosine similarity
        sim = torch.nn.functional.cosine_similarity(emb_q, mean_emb, dim=0).item()
        if sim > best_score:
            best_score = sim
            best_label = lbl

    return f"Predicted: {best_label} (cosine similarity={best_score:.3f})"


def launch_app(capture_dir: Path, device: torch.device):
    ensure_dirs(capture_dir)
    facenet = build_facenet(device)

    with gr.Blocks() as demo:
        gr.Markdown("# OAK-D Capture + Label + ID (FaceNet embeddings)")

        with gr.Tab("Capture"):
            capture_btn = gr.Button("Capture from OAK-D")
            capture_status = gr.Textbox(label="Status", interactive=False)
            gallery = gr.Gallery(label="Captures", value=list_images(capture_dir), columns=4, height=300)

            def do_capture():
                path = capture_from_oakd(capture_dir)
                return f"Saved {path}", list_images(capture_dir)

            capture_btn.click(fn=do_capture, outputs=[capture_status, gallery])

        with gr.Tab("Label"):
            gallery_sel = gr.Gallery(label="Select images to label", value=list_images(capture_dir), selectable=True, columns=4, height=300)
            label_text = gr.Textbox(label="Label (e.g., John Doe)")
            label_btn = gr.Button("Apply label")
            label_status = gr.Textbox(label="Status", interactive=False)

            def do_label(selected, label):
                selected = selected or []
                if not label:
                    return "Provide a label."
                return label_images(selected, label)

            label_btn.click(fn=do_label, inputs=[gallery_sel, label_text], outputs=label_status)

        with gr.Tab("Identify"):
            query_img = gr.Image(type="filepath", label="Upload or pick a capture", height=256)
            identify_btn = gr.Button("Identify")
            identify_out = gr.Textbox(label="Prediction", interactive=False)

            identify_btn.click(
                fn=lambda img_path: identify_image(img_path, device, facenet),
                inputs=query_img,
                outputs=identify_out,
            )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


def parse_args():
    ap = argparse.ArgumentParser(description="OAK-D capture/label/ID UI")
    ap.add_argument("--capture-dir", type=str, default=DEFAULT_CAPTURE_DIR.as_posix())
    ap.add_argument("--device", type=str, default="cpu", help="Device for FaceNet inference (cpu/cuda)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    launch_app(Path(args.capture_dir), torch.device(args.device))
