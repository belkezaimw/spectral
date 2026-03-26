"""
train.py
========
Full training loop with AMP, checkpointing, TensorBoard, and LR scheduling.

Quick start:
    uv run gf-train --train_dirs data/DIV2K/DIV2K_train_HR --val_dirs data/DIV2K/DIV2K_valid_HR
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from guided_fusion.dataset import build_dataloaders
from guided_fusion.losses import GuidedFusionLoss
from guided_fusion.metrics import psnr, ssim
from guided_fusion.model import GuidedFusionNet, count_parameters


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_psnr   = ckpt.get("best_psnr", 0.0)
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"[Train] Resumed from epoch {ckpt.get('epoch', 0)}  (best PSNR: {best_psnr:.2f} dB)")
    return start_epoch, best_psnr


# ─────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch, writer):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for i, (gray, lq, hq) in enumerate(tqdm(loader, desc=f"Epoch {epoch}", leave=False)):
        gray = gray.to(device, non_blocking=True)   # [B, 1, H, W]
        lq   = lq.to(device, non_blocking=True)     # [B, 3, H, W]
        hq   = hq.to(device, non_blocking=True)     # [B, 3, H, W]

        optimizer.zero_grad()

        with autocast(enabled=scaler.is_enabled()):
            pred   = model(lq, gray)
            losses = criterion(pred, hq)
            loss   = losses["total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    elapsed  = time.time() - t0
    writer.add_scalar("train/loss", avg_loss, epoch)
    print(f"[Epoch {epoch:03d}] train_loss={avg_loss:.4f}  ({elapsed:.1f}s)")
    return avg_loss


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device, epoch, writer):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0

    for gray, lq, hq in tqdm(loader, desc="  Val", leave=False):
        gray = gray.to(device)
        lq   = lq.to(device)
        hq   = hq.to(device)
        pred = model(lq, gray).clamp(0, 1)
        total_psnr += psnr(pred, hq).item()
        total_ssim += ssim(pred, hq).item()

    n      = len(loader)
    avg_p  = total_psnr / n
    avg_s  = total_ssim / n
    writer.add_scalar("val/psnr", avg_p, epoch)
    writer.add_scalar("val/ssim", avg_s, epoch)
    print(f"          val_psnr={avg_p:.2f} dB   val_ssim={avg_s:.4f}")
    return avg_p, avg_s


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train GuidedFusionNet")

    # Data
    parser.add_argument("--train_dirs", nargs="+", required=True,
                        help="Directories with HQ training images (e.g. data/DIV2K/DIV2K_train_HR)")
    parser.add_argument("--val_dirs",   nargs="+", required=True,
                        help="Directories with HQ validation images")
    parser.add_argument("--crop_size",  type=int,   default=256)
    parser.add_argument("--scale",      type=int,   default=4,
                        help="LQ downscale factor (2 or 4)")
    parser.add_argument("--noise_std",  type=float, default=25.0,
                        help="Noise level added to LQ (0-255 scale)")

    # Model
    parser.add_argument("--base_ch",    type=int,   default=32,
                        help="Base channels (32=~3M params, 64=~11M params)")

    # Training
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--num_workers",type=int,   default=4)
    parser.add_argument("--amp",        action="store_true", help="Use mixed precision")

    # Checkpointing
    parser.add_argument("--save_dir",   type=str,   default="./checkpoints")
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--val_every",  type=int,   default=5,
                        help="Validate every N epochs")

    args = parser.parse_args()

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    if device.type == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──
    train_loader, val_loader = build_dataloaders(
        train_roots  = args.train_dirs,
        val_roots    = args.val_dirs,
        crop_size    = args.crop_size,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        scale        = args.scale,
        noise_std    = args.noise_std,
    )
    print(f"[Train] Batches: {len(train_loader)} train / {len(val_loader)} val")

    # ── Model ──
    model = GuidedFusionNet(base_ch=args.base_ch).to(device)
    print(f"[Train] Parameters: {count_parameters(model):,}")

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler(enabled=args.amp)
    criterion = GuidedFusionLoss()

    # ── Resume ──
    start_epoch = 1
    best_psnr   = 0.0
    save_dir    = Path(args.save_dir)

    if args.resume and Path(args.resume).exists():
        start_epoch, best_psnr = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler
        )

    # ── TensorBoard ──
    writer = SummaryWriter(log_dir=str(save_dir / "logs"))

    # Save config
    (save_dir / "config.json").parent.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Training loop ──
    print(f"[Train] Starting from epoch {start_epoch} → {args.epochs}")
    for epoch in range(start_epoch, args.epochs + 1):

        train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, writer)
        scheduler.step()

        # Checkpoint every epoch
        save_checkpoint({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_psnr": best_psnr,
        }, save_dir / "last.pth")

        # Validate periodically
        if epoch % args.val_every == 0 or epoch == args.epochs:
            avg_psnr, avg_ssim = validate(model, val_loader, device, epoch, writer)

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                save_checkpoint({
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "best_psnr": best_psnr,
                }, save_dir / "best_model.pth")
                print(f"          ✓ New best PSNR: {best_psnr:.2f} dB — saved best_model.pth")

    writer.close()
    print(f"\n[Train] Done. Best PSNR: {best_psnr:.2f} dB")
    print(f"[Train] Best model: {save_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
