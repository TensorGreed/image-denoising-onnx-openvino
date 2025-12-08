# image-denoising-onnx-openvino

Minimal setup to train a face denoiser with a HIP-based noise op (tuned for MI300X), then export the trained model to ONNX → OpenVINO IR → optional OAK-D blob, and publish checkpoints to Hugging Face.

## Components
- `hip_addnoise/`: HIP extension to add Gaussian noise on ROCm (supports fp32/fp16/bf16).
- `train_face_denoiser.py`: trains a FaceDenoiser autoencoder with identity loss (FaceNet) on LFW; uses HIP noise augmentation.
- `export_to_openvino.py`: loads a checkpoint, exports ONNX, converts to OpenVINO IR (FP16), optionally builds an OAK-D blob via `blobconverter`.
- `train_upload_export.sh`: one-shot script to train → upload to Hugging Face → export ONNX/IR/(blob).

## Quickstart
1) Install deps (examples):
   ```bash
   # Base: ROCm-ready PyTorch on MI300X (adjust rocm version if needed)
   pip install --upgrade pip
   pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1

   # Python deps
   pip install facenet-pytorch openvino blobconverter huggingface_hub

   # Local HIP extensions
   python -m pip install -e .
   ```
   Ensure ROCm/CUDA stack is installed for MI300X; for OAK-D export, OpenVINO and `blobconverter` must be present.

2) Train:
   ```bash
   python train_face_denoiser.py \
     --data-dir ./data_lfw \
     --output-dir ./outputs_face \
     --batch-size 32 --epochs 5 --lr 1e-4 \
     --noise-std 0.1 --num-workers 4 --latent-dim 512 --lambda-id 0.1
   ```

3) Export (ONNX + IR, optional blob):
   ```bash
   python export_to_openvino.py \
     --ckpt ./outputs_face/face_denoiser_best_epochX_vallossY.pt \
     --output-dir ./exported \
     --opset 17 --device cpu \
     --export-blob --blob-shaves 6 --openvino-version 2022.1.0
   ```

4) One-shot pipeline (train → HF upload → export):
   ```bash
   HF_TOKEN=your_token HF_REPO=your-username/face-denoiser-mi300x \
   EXPORT_BLOB=true BLOB_SHAVES=6 OPENVINO_VERSION=2022.1.0 \
   bash train_upload_export.sh
   ```
   - Requires `huggingface-cli` in PATH; uploads latest checkpoint to `HF_REPO` under `checkpoints/`.

## Notes for MI300X
- HIP noise kernel supports fp32/fp16/bf16 and uses 512 threads/block; adjust if benchmarking suggests otherwise.
- Add backward if you plan to train with learnable noise or need autograd through the op.

## Notes for OAK-D
- OAK-D cannot run HIP; keep the noise op in training only. Exported ONNX/IR should be static input shape (B×3×112×112).
- Use the generated blob with DepthAI pipelines; match `--openvino-version` to device firmware expectations.
