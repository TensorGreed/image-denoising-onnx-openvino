#!/usr/bin/env bash
# Train FaceDenoiser -> upload checkpoint to Hugging Face -> export to ONNX/OpenVINO/OAK-D blob.
# Requirements: python deps for training, huggingface_hub/huggingface-cli installed, OpenVINO + blobconverter for export.
# Env vars needed: HF_TOKEN (write token), HF_REPO (e.g., your-username/face-denoiser-mi300x)

set -euo pipefail

# -----------------------------
# Configurable parameters
# -----------------------------
OUTPUT_DIR="${OUTPUT_DIR:-./outputs_face}"
DATA_DIR="${DATA_DIR:-./data_lfw}"
EXPORT_DIR="${EXPORT_DIR:-./exported}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-4}"
NOISE_STD="${NOISE_STD:-0.1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LATENT_DIM="${LATENT_DIM:-256}"
LAMBDA_ID="${LAMBDA_ID:-0.1}"
EXPORT_OPSET="${EXPORT_OPSET:-17}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"      # use "cuda" if you have ROCm/CUDA available
EXPORT_BLOB="${EXPORT_BLOB:-false}"        # set to "true" to also build OAK-D blob
BLOB_SHAVES="${BLOB_SHAVES:-6}"
OPENVINO_VERSION="${OPENVINO_VERSION:-2022.1.0}"
IMAGE_SIZE="${IMAGE_SIZE:-96}"

# -----------------------------
# Checks
# -----------------------------
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN env var not set." >&2
  exit 1
fi
if [[ -z "${HF_REPO:-}" ]]; then
  echo "ERROR: HF_REPO env var not set (e.g., your-username/face-denoiser-mi300x)." >&2
  exit 1
fi

# -----------------------------
# 1) Train
# -----------------------------
echo "[1/3] Training FaceDenoiser -> ${OUTPUT_DIR}"
python train_face_denoiser.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --image-size "${IMAGE_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --noise-std "${NOISE_STD}" \
  --num-workers "${NUM_WORKERS}" \
  --latent-dim "${LATENT_DIM}" \
  --lambda-id "${LAMBDA_ID}"

LATEST_CKPT=$(ls -1t "${OUTPUT_DIR}"/*.pt | head -1)
if [[ -z "${LATEST_CKPT}" ]]; then
  echo "ERROR: No checkpoint found in ${OUTPUT_DIR}" >&2
  exit 1
fi
echo "Latest checkpoint: ${LATEST_CKPT}"

# -----------------------------
# 2) Upload to Hugging Face
# -----------------------------
echo "[2/3] Uploading to Hugging Face: repo=${HF_REPO}"
huggingface-cli upload "${HF_REPO}" "${LATEST_CKPT}" "checkpoints/$(basename "${LATEST_CKPT}")" \
  --repo-type model \
  --token "${HF_TOKEN}" \
  --create-pr 0

# -----------------------------
# 3) Export ONNX -> OpenVINO -> (optional) OAK-D blob
# -----------------------------
echo "[3/3] Exporting to ONNX/OpenVINO at ${EXPORT_DIR}"
EXPORT_ARGS=(
  --ckpt "${LATEST_CKPT}"
  --output-dir "${EXPORT_DIR}"
  --image-size "${IMAGE_SIZE}"
  --opset "${EXPORT_OPSET}"
  --device "${EXPORT_DEVICE}"
)
if [[ "${EXPORT_BLOB}" == "true" ]]; then
  EXPORT_ARGS+=(--export-blob --blob-shaves "${BLOB_SHAVES}" --openvino-version "${OPENVINO_VERSION}")
fi

python export_to_openvino.py "${EXPORT_ARGS[@]}"

echo "Done."
