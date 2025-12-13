# hf_upload.py
"""
Upload a checkpoint to Hugging Face using huggingface_hub.

Usage:
  HF_TOKEN=... python hf_upload.py --repo your-username/face-denoiser-mi300x \
    --file outputs_face/face_denoiser_best_epoch5_valloss0.0485.pt \
    --dest checkpoints/face_denoiser_best.pt
"""

import argparse
import os

from huggingface_hub import HfApi


def parse_args():
    p = argparse.ArgumentParser(description="Upload a file to Hugging Face model repo")
    p.add_argument("--repo", required=True, help="Repo ID, e.g., your-username/model-name")
    p.add_argument("--file", required=True, help="Local file path to upload")
    p.add_argument("--dest", required=True, help="Destination path in repo, e.g., checkpoints/file.pt")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (or set HF_TOKEN env)")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.token:
        raise SystemExit("HF token not provided. Set --token or HF_TOKEN env var.")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=args.file,
        path_in_repo=args.dest,
        repo_id=args.repo,
        repo_type="model",
        token=args.token,
    )
    print(f"Uploaded {args.file} to {args.repo}:{args.dest}")


if __name__ == "__main__":
    main()
