# image-denoising-onnx-openvino

Minimal setup to train a face denoiser with a HIP-based noise op (tuned for MI300X), export to ONNX -> OpenVINO IR -> optional OAK-D blob, and publish checkpoints to Hugging Face.

## Components
- `hip_addnoise/`: HIP extension to add Gaussian noise on ROCm (fp32).
- `hip_linear/`: HIP/hipBLAS linear op (optional, fp32).
- `train_face_denoiser.py`: trains a FaceDenoiser autoencoder with identity loss (FaceNet) on LFW; uses HIP noise augmentation.
- `export_to_openvino.py`: loads a checkpoint, exports ONNX, converts to OpenVINO IR (FP16), optionally builds an OAK-D blob via `blobconverter`.
- `train_upload_export.sh`: one-shot script to train -> upload to Hugging Face -> export ONNX/IR/(blob).
- `oakd_face_label_app.py`: simple host UI to capture from OAK-D, label, and identify faces via embeddings.

## Quickstart
1) Install deps (examples):
   ```bash
   # Base: ROCm-ready PyTorch on MI300X (adjust rocm version if needed)
   pip install --upgrade pip
   pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1

   # Python deps
   pip install facenet-pytorch openvino blobconverter huggingface_hub

   # Local HIP extensions (install from subfolders)
   # Ensure ROCm toolchain is visible
   export ROCM_HOME=${ROCM_HOME:-/opt/rocm}
   export CUDA_HOME=${CUDA_HOME:-$ROCM_HOME}
   export CUDACXX=${CUDACXX:-$ROCM_HOME/bin/hipcc}
   export PATH="$ROCM_HOME/bin:$PATH"
   # Set arch for MI300X
   export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-gfx942}
   python -m pip install -e hip_addnoise --no-build-isolation
   python -m pip install -e hip_linear --no-build-isolation  # optional, for the hipBLAS linear op
   ```
   Ensure ROCm stack is installed on MI300X; for OAK-D export, OpenVINO and `blobconverter` must be present.

2) Train (OAK-D-friendly defaults: 96x96 input, slim model):
   ```bash
   python train_face_denoiser.py \
     --data-dir ./data_lfw \
     --output-dir ./outputs_face \
     --image-size 96 \
     --batch-size 16 --epochs 5 --lr 1e-4 \
     --noise-std 0.1 --num-workers 4 --latent-dim 256 --lambda-id 0.1
   ```

3) Export (ONNX + IR, optional blob):
   ```bash
   python export_to_openvino.py \
     --ckpt ./outputs_face/face_denoiser_best_epochX_vallossY.pt \
     --output-dir ./exported \
     --image-size 96 \
     --opset 17 --device cpu \
     --export-blob --blob-shaves 6 --openvino-version 2022.1.0
   ```

4) One-shot pipeline (train -> HF upload -> export):
   ```bash
   HF_TOKEN=your_token HF_REPO=your-username/face-denoiser-mi300x \
   IMAGE_SIZE=96 \
   EXPORT_BLOB=true BLOB_SHAVES=6 OPENVINO_VERSION=2022.1.0 \
   bash train_upload_export.sh
   ```
   Requires `huggingface-cli` in PATH; uploads latest checkpoint to `HF_REPO` under `checkpoints/`.

## Dataset note (LFW and CelebA)
- LFW: TorchVision no longer auto-downloads LFWPeople. Download and extract the “lfw-funneled” archive plus split files (peopleDevTrain.txt / peopleDevTest.txt). If you have the Kaggle LFW archive, extract `lfw-funneled.tgz` into `--data-dir` and place the split files there. For `--data-dir ./data`, you should end up with:
  - `./data/lfw_funneled/...`
  - `./data/peopleDevTrain.txt`
  - `./data/peopleDevTest.txt` (pairs*.txt optional)
- CelebA: Use the manually downloaded CelebA files and keep them under `--data-dir`. Expected layout:
  - `./data/img_align_celeba/000001.jpg` ... (all images)
  - `./data/list_eval_partition.csv` (code uses the .csv directly)
  - `./data/list_attr_celeba.csv`
  - `./data/list_bbox_celeba.csv`
  - `./data/list_landmarks_align_celeba.csv`
- Both: Set `--dataset both` to concatenate LFW and CelebA (requires both present under `--data-dir`).

## Notes for MI300X
- HIP noise kernel supports fp32 and uses 512 threads/block; adjust if benchmarking suggests otherwise.
- Add backward if you plan to train with learnable noise or need autograd through the op.

## Notes for OAK-D
- OAK-D cannot run HIP; keep the noise op in training only. Exported ONNX/IR should be static input shape (B x 3 x H x W), default 96 x 96 here.
- Steps to run on OAK-D:
  1) Export blob: `python export_to_openvino.py --ckpt ./outputs_face/best.pt --output-dir ./exported --image-size 96 --export-blob --blob-shaves 6 --openvino-version 2022.1.0`
  2) In your DepthAI script, load the blob: `nn.setBlobPath("exported/oakd/face_denoiser.blob")` and feed 3x96x96 RGB inputs.
  3) Keep preprocessing consistent with training (resize to 96x96, normalize to [0,1]).
- Match `--openvino-version` to your DepthAI firmware; regenerate the blob if versions differ.

## Simple OAK-D UI (capture -> label -> identify)
- File: `oakd_face_label_app.py`
- What it does: captures RGB frames from OAK-D, lets you label images (e.g., "John Doe"), and identifies a query image by comparing FaceNet embeddings to labeled gallery.
- Install extras:
  ```bash
  pip install gradio facenet-pytorch depthai opencv-python torchvision torch --index-url https://download.pytorch.org/whl/rocm6.1
  ```
  Adjust the torch index-url as needed for your ROCm/CUDA setup.
- Run on a host with the OAK-D connected over USB:
  ```bash
  python oakd_face_label_app.py --capture-dir ./oakd_captures --device cpu
  ```
  Then open the printed URL (default http://0.0.0.0:7860) to capture, label, and identify. Use `--device cuda` if you want embeddings on GPU. If headless, port-forward (e.g., `ssh -L 7860:localhost:7860 user@host`) and open http://localhost:7860.

Notes:
- Embedding runs on host (CPU/GPU); not on the OAK-D VPU. You can swap FaceNet for an OpenVINO face-recognition model if desired.
- Labeled images and labels JSON are stored under `./oakd_captures` by default.
