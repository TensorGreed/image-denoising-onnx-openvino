# train_face_denoiser.py

import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import hip_addnoise
from facenet_pytorch import InceptionResnetV1


# ---------------------------
# Face Denoising Autoencoder + Latent MLP
# OAK-D friendly: smaller channels/resolution defaults (96x96), lightweight decoder.
# ---------------------------

class FaceDenoiser(nn.Module):
    def __init__(self, latent_dim: int = 256, image_size: int = 96):
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16 for the current encoder/decoder strides.")

        # Encoder: slimmed conv stack
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),   # [B,24,H/2,W/2]
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),  # [B,48,H/4,W/4]
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),  # [B,96,H/8,W/8]
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1), # [B,128,H/16,W/16]
            nn.ReLU(inplace=True),
        )

        # Feature size after encoder
        self.enc_feat_c = 128
        self.enc_feat_h = image_size // 16
        self.enc_feat_w = image_size // 16
        enc_feat_dim = self.enc_feat_c * self.enc_feat_h * self.enc_feat_w

        # Latent bottleneck MLP (this is where hip_linear could be used in inference)
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(enc_feat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, enc_feat_dim)

        # Decoder: mirror-ish upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 96, kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),  # [B,96,H/8,W/8]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                96, 48, kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),  # [B,48,H/4,W/4]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                48, 24, kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),  # [B,24,H/2,W/2]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                24, 16, kernel_size=3, stride=2,
                padding=1, output_padding=1
            ),  # [B,16,H,W]
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),  # [B,3,H,W]
            nn.Sigmoid(),  # outputs in [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)          # [B,256,7,7]
        z_flat = self.flatten(z)     # [B, 256*7*7]
        z_latent = self.fc_enc(z_flat)
        z_recon = self.fc_dec(z_latent)
        z_recon = z_recon.view(-1, self.enc_feat_c, self.enc_feat_h, self.enc_feat_w)
        out = self.decoder(z_recon)  # [B,3,H,W]
        return out


# ---------------------------
# Datasets: LFW faces with on-the-fly noise
# ---------------------------

def get_lfw_dataloaders(data_dir, batch_size, num_workers, image_size=96, noise_std=0.1):
    """
    Returns train and val dataloaders for LFWPeople, resized to `image_size`.
    LFW auto-downloads are disabled upstream; ensure `lfw_funneled` and split files
    (peopleDevTrain.txt, peopleDevTest.txt or pairs*.txt) exist under `data_dir`.
    """

    transform_clean = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),          # [0,1], shape [3,H,W]
    ])

    # LFWPeople returns (image, label). We'll add noise in the training loop using hip_addnoise.

    # LFWPeople expects peopleDevTrain/peopleDevTest; if you only have pairs*.txt,
    # we can still use train/test splits by reusing peopleDevTrain/peopleDevTest naming.
    train_ds = datasets.LFWPeople(
        root=data_dir,
        split="train",
        image_set="funneled",
        transform=transform_clean,
        download=False,
    )

    val_ds = datasets.LFWPeople(
        root=data_dir,
        split="test",    # use test as val for now
        image_set="funneled",
        transform=transform_clean,
        download=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_celeba_dataloaders(data_dir, batch_size, num_workers, image_size=96):
    """
    Returns train and val dataloaders for CelebA. Requires manual download to data_dir.
    """
    transform_clean = transforms.Compose([
        transforms.CenterCrop(178),  # CelebA default
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.CelebA(
        root=data_dir,
        split="train",
        transform=transform_clean,
        target_type="attr",
        partition_file="list_eval_partition.csv",
        download=False,
    )

    val_ds = datasets.CelebA(
        root=data_dir,
        split="valid",
        transform=transform_clean,
        target_type="attr",
        partition_file="list_eval_partition.csv",
        download=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_mixed_dataloaders(data_dir, batch_size, num_workers, image_size=96, noise_std=0.1):
    """
    Returns combined train/val loaders by mixing LFW and CelebA.
    Assumes both datasets are present under data_dir.
    """
    lfw_train, lfw_val = get_lfw_dataloaders(
        data_dir, batch_size, num_workers, image_size=image_size, noise_std=noise_std
    )
    celeba_train, celeba_val = get_celeba_dataloaders(
        data_dir, batch_size, num_workers, image_size=image_size
    )

    # Combine datasets with concatenation; use weighted batch sampling implicitly via concatenation.
    train_ds = torch.utils.data.ConcatDataset([lfw_train.dataset, celeba_train.dataset])
    val_ds = torch.utils.data.ConcatDataset([lfw_val.dataset, celeba_val.dataset])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# ---------------------------
# Identity model (FaceNet / InceptionResnetV1)
# ---------------------------

def build_facenet(device):
    """
    Pretrained InceptionResnetV1 returning embeddings, not a classifier.
    pretrained='vggface2' uses model trained on VGGFace2. 
    """
    facenet = InceptionResnetV1(pretrained="vggface2", classify=False)
    facenet.eval()
    facenet.to(device)
    for p in facenet.parameters():
        p.requires_grad = False
    return facenet


# ---------------------------
# Train / Eval epoch
# ---------------------------

def compute_identity_loss(facenet, clean, denoised):
    """
    Compute identity loss between clean and denoised faces
    using FaceNet embeddings (L2 distance).
    Inputs:
        clean, denoised: [B,3,H,W] float in [0,1]
    """
    with torch.no_grad():
        emb_clean = facenet(clean)       # [B,512]
    emb_denoised = facenet(denoised)     # [B,512]

    # Mean L2 distance
    return ((emb_clean - emb_denoised) ** 2).sum(dim=1).mean()


def train_epoch(model, facenet, loader, optimizer, device, noise_std, lambda_id):
    model.train()
    running_loss = 0.0

    pixel_criterion = nn.MSELoss()

    for clean, _ in loader:
        clean = clean.to(device, non_blocking=True)  # [B,3,H,W]

        # Add noise with HIP kernel (same pattern as before)
        noisy = hip_addnoise.add_noise(clean, noise_std=noise_std)

        optimizer.zero_grad()

        denoised = model(noisy)

        # Pixel loss
        pixel_loss = pixel_criterion(denoised, clean)

        # Identity loss
        id_loss = compute_identity_loss(facenet, clean, denoised)

        loss = pixel_loss + lambda_id * id_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * clean.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, facenet, loader, device, noise_std, lambda_id):
    model.eval()
    running_loss = 0.0

    pixel_criterion = nn.MSELoss()

    for clean, _ in loader:
        clean = clean.to(device, non_blocking=True)
        noisy = hip_addnoise.add_noise(clean, noise_std=noise_std)
        denoised = model(noisy)

        pixel_loss = pixel_criterion(denoised, clean)
        id_loss = compute_identity_loss(facenet, clean, denoised)
        loss = pixel_loss + lambda_id * id_loss

        running_loss += loss.item() * clean.size(0)

    return running_loss / len(loader.dataset)


# ---------------------------
# Argument parsing + main
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Face denoiser (LFW) on ROCm/MI300X with HIP noise + FaceNet identity loss"
    )

    p.add_argument("--data-dir", type=str, default="./data_lfw",
                   help="Where LFW will be downloaded")
    p.add_argument("--dataset", type=str, default="lfw", choices=["lfw", "celeba", "both"],
                   help="Which dataset to use")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--noise-std", type=float, default=0.1,
                   help="Std dev of Gaussian noise added by HIP kernel")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="./outputs_face")
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--lambda-id", type=float, default=0.1,
                   help="Weight for identity loss vs pixel loss")
    p.add_argument("--image-size", type=int, default=96,
                   help="Input resolution H=W for training/export")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==========================================")
    print("Device:", device)
    if device.type == "cuda":
        print("CUDA/ROCm device count:", torch.cuda.device_count())
        print("Using:", torch.cuda.get_device_name(0))
    print("==========================================")

    # Data
    if args.dataset == "lfw":
        train_loader, val_loader = get_lfw_dataloaders(
            args.data_dir, args.batch_size, args.num_workers,
            image_size=args.image_size, noise_std=args.noise_std  # noise_std passed to train/eval, not here
        )
    elif args.dataset == "celeba":
        train_loader, val_loader = get_celeba_dataloaders(
            args.data_dir, args.batch_size, args.num_workers,
            image_size=args.image_size
        )
    elif args.dataset == "both":
        train_loader, val_loader = get_mixed_dataloaders(
            args.data_dir, args.batch_size, args.num_workers,
            image_size=args.image_size, noise_std=args.noise_std
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Models
    model = FaceDenoiser(latent_dim=args.latent_dim, image_size=args.image_size).to(device)
    facenet = build_facenet(device)

    print("Denoiser params:",
          sum(p.numel() for p in model.parameters()) / 1e6, "M")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, facenet, train_loader, optimizer,
            device, args.noise_std, args.lambda_id
        )
        val_loss = eval_epoch(
            model, facenet, val_loader,
            device, args.noise_std, args.lambda_id
        )

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(
                args.output_dir,
                f"face_denoiser_best_epoch{epoch}_valloss{val_loss:.4f}.pt",
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "timestamp": datetime.utcnow().isoformat(),
                    "latent_dim": args.latent_dim,
                },
                ckpt_path,
            )
            print(f"  ↳ New best model saved to {ckpt_path}")

    print("Training finished. Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()
