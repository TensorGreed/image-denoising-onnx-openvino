# hf_download_and_export.py
"""
Download a checkpoint from Hugging Face and export to ONNX/OpenVINO/blob.

Usage:
  HF_TOKEN=... python hf_download_and_export.py \
    --repo your-username/face-denoiser-mi300x \
    --path-in-repo checkpoints/face_denoiser_best.pt \
    --output-dir ./exported \
    --image-size 96 \
    --export-blob --blob-shaves 6 --openvino-version 2022.1.0
"""

import argparse
import os
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args():
    p = argparse.ArgumentParser(description="Download HF checkpoint and export to ONNX/OpenVINO/blob")
    p.add_argument("--repo", required=True, help="Repo ID, e.g., your-username/model-name")
    p.add_argument("--path-in-repo", required=True, help="Path inside repo to checkpoint, e.g., checkpoints/file.pt")
    p.add_argument("--output-dir", type=str, default="./exported", help="Export output directory")
    p.add_argument("--image-size", type=int, default=96, help="Input H=W for export")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p.add_argument("--device", type=str, default="cpu", help="Device for export")
    p.add_argument("--export-blob", action="store_true", help="Also build OAK-D blob")
    p.add_argument("--blob-shaves", type=int, default=6, help="Blob shaves for OAK-D")
    p.add_argument("--openvino-version", type=str, default="2022.1.0", help="OpenVINO version for blobconverter")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token or set HF_TOKEN env")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.token:
        raise SystemExit("HF token not provided. Set --token or HF_TOKEN env var.")

    local_dir = Path("./hf_download")
    # Download the snapshot; this pulls the whole repo (cached), then we pick the file we need.
    snapshot_path = snapshot_download(
        repo_id=args.repo,
        repo_type="model",
        token=args.token,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=args.path_in_repo,
    )

    ckpt_path = Path(snapshot_path) / args.path_in_repo
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found after download: {ckpt_path}")

    # Build args for export_to_openvino CLI and invoke via subprocess
    export_args = [
        "python", "export_to_openvino.py",
        "--ckpt", ckpt_path.as_posix(),
        "--output-dir", args.output_dir,
        "--image-size", str(args.image_size),
        "--opset", str(args.opset),
        "--device", args.device,
    ]
    if args.export_blob:
        export_args.extend([
            "--export-blob",
            "--blob-shaves", str(args.blob_shaves),
            "--openvino-version", args.openvino_version,
        ])

    subprocess.run(export_args, check=True)


if __name__ == "__main__":
    main()
