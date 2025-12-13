# export_to_openvino.py
# Export FaceDenoiser checkpoints to ONNX, OpenVINO IR, and optionally OAK-D blob.
import argparse
import os
from pathlib import Path
from typing import Tuple

import torch

from train_face_denoiser import FaceDenoiser


def load_model(ckpt_path: Path, device: torch.device) -> FaceDenoiser:
    state = torch.load(ckpt_path, map_location=device)
    latent_dim = state.get("latent_dim", 512)
    model = FaceDenoiser(latent_dim=latent_dim)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    model.to(device)
    return model


def export_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    onnx_path: Path,
    opset: int,
) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["denoised"],
        dynamic_axes=None,  # keep static shape for OAK-D
    )


def export_openvino_ir(
    onnx_path: Path,
    ir_dir: Path,
    compress_to_fp16: bool = True,
) -> Tuple[Path, Path]:
    from openvino.tools import mo
    from openvino.runtime import serialize

    ir_dir.mkdir(parents=True, exist_ok=True)
    model_ir = mo.convert_model(onnx_path.as_posix(), compress_to_fp16=compress_to_fp16)
    # Avoid extra dots in filename (blobconverter rejects them)
    suffix = "_fp16" if compress_to_fp16 else ""
    xml_path = ir_dir / f"{onnx_path.stem}{suffix}.xml"
    bin_path = xml_path.with_suffix(".bin")
    serialize(model_ir, xml_path.as_posix(), bin_path.as_posix())
    return xml_path, bin_path


def export_oakd_blob(
    xml_path: Path,
    bin_path: Path,
    blob_path: Path,
    shaves: int,
    openvino_version: str,
) -> Path:
    try:
        import blobconverter
    except ImportError as e:
        raise RuntimeError(
            "blobconverter not installed. Install with `pip install blobconverter`."
        ) from e

    blob_path.parent.mkdir(parents=True, exist_ok=True)
    # DepthAI/Blobconverter expects OpenVINO version matching device firmware, e.g., 2022.1.0
    result_path = blobconverter.from_openvino(
        xml=xml_path.as_posix(),
        bin=bin_path.as_posix(),
        data_type="FP16",
        shaves=shaves,
        version=openvino_version,
        output_dir=blob_path.parent.as_posix(),
        output_blob_name=blob_path.name,
    )
    return Path(result_path)


def parse_args():
    p = argparse.ArgumentParser(
        description="Export FaceDenoiser checkpoint to ONNX -> OpenVINO IR -> OAK-D blob"
    )
    p.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument("--output-dir", type=str, default="./exported",
                   help="Base directory for exports")
    p.add_argument("--image-size", type=int, default=96,
                   help="Input H=W resolution expected by the model")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size for export (static)")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for export (cpu or cuda)")
    p.add_argument("--no-ir", action="store_true",
                   help="Skip OpenVINO IR export")
    p.add_argument("--export-blob", action="store_true",
                   help="Also export OAK-D blob (requires blobconverter)")
    p.add_argument("--blob-shaves", type=int, default=6,
                   help="Number of SHAVEs to target for MyriadX blob")
    p.add_argument("--openvino-version", type=str, default="2022.1.0",
                   help="OpenVINO version string for blobconverter (must match DepthAI firmware)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.output_dir)
    onnx_path = out_dir / "face_denoiser.onnx"
    ir_dir = out_dir / "openvino_ir"
    blob_path = out_dir / "oakd" / "face_denoiser.blob"

    model = load_model(ckpt_path, device)

    dummy = torch.zeros(
        (args.batch_size, 3, args.image_size, args.image_size),
        device=device,
        dtype=torch.float32,
    )

    print(f"[1/3] Exporting ONNX to {onnx_path}")
    export_onnx(model, dummy, onnx_path, args.opset)

    xml_path = bin_path = None
    if not args.no_ir:
        print(f"[2/3] Converting ONNX -> OpenVINO IR at {ir_dir}")
        xml_path, bin_path = export_openvino_ir(onnx_path, ir_dir, compress_to_fp16=True)
        print(f"      IR written: {xml_path}, {bin_path}")
    else:
        print("[2/3] Skipping IR export (--no-ir)")

    if args.export_blob:
        if xml_path is None or bin_path is None:
            raise RuntimeError("Blob export requested but IR was not generated.")
        print(f"[3/3] Building OAK-D blob at {blob_path}")
        blob_out = export_oakd_blob(
            xml_path, bin_path, blob_path, args.blob_shaves, args.openvino_version
        )
        print(f"      Blob written: {blob_out}")
    else:
        print("[3/3] Skipping blob export (--export-blob to enable)")

    print("Export complete.")


if __name__ == "__main__":
    main()
