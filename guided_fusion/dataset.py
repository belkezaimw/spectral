"""
dataset.py
==========
PyTorch Dataset that generates training pairs on-the-fly from high-quality images.

From each high-quality color image we derive:
  - HQ Grayscale  : luminance channel, full resolution, sharp        (input 1)
  - LQ RGB        : downscaled + noise + blur, then upscaled back     (input 2)
  - HQ Color      : original color image                              (target)

Supported datasets (place under data/):
  data/
    DIV2K/
      DIV2K_train_HR/   ← 800 images, download from DIV2K website
      DIV2K_valid_HR/   ← 100 images
    Flickr2K/
      Flickr2K_HR/      ← 2650 images (optional, for more data)
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────
# Degradation pipeline  (LQ RGB synthesis)
# ─────────────────────────────────────────────────────────────

def degrade_rgb(img_hq: np.ndarray, scale: int = 4, noise_std: float = 25.0) -> np.ndarray:
    """
    Synthesize a low-quality RGB from a high-quality color image.

    Steps:
      1. Gaussian blur  (anti-aliasing before downscale)
      2. Downsample  ×scale
      3. Add Gaussian noise
      4. Upsample back to original size  (bicubic)

    Args:
        img_hq    : HxWx3 float32 in [0, 1]
        scale     : downscale factor (2 or 4)
        noise_std : noise level in [0, 255] scale
    Returns:
        LQ image  : HxWx3 float32 in [0, 1]
    """
    h, w = img_hq.shape[:2]

    # 1. Blur
    k = 2 * scale - 1
    blurred = cv2.GaussianBlur(img_hq, (k, k), sigmaX=scale * 0.6)

    # 2. Downsample
    lq_small = cv2.resize(blurred, (w // scale, h // scale), interpolation=cv2.INTER_AREA)

    # 3. Noise
    noise = np.random.normal(0, noise_std / 255.0, lq_small.shape).astype(np.float32)
    lq_small = np.clip(lq_small + noise, 0.0, 1.0)

    # 4. Upsample back to HQ size
    lq = cv2.resize(lq_small, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(lq, 0.0, 1.0)


def rgb_to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert HxWx3 float32 to HxW float32 using luminance weights."""
    return (0.2126 * img[:, :, 0] +
            0.7152 * img[:, :, 1] +
            0.0722 * img[:, :, 2]).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Augmentations
# ─────────────────────────────────────────────────────────────

def random_crop(gray: np.ndarray, lq: np.ndarray, hq: np.ndarray, size: int):
    h, w = gray.shape[:2]
    top  = random.randint(0, h - size)
    left = random.randint(0, w - size)
    s = np.s_[top:top+size, left:left+size]
    return gray[s], lq[s], hq[s]


def random_flip(gray: np.ndarray, lq: np.ndarray, hq: np.ndarray):
    if random.random() > 0.5:
        gray = np.fliplr(gray).copy()
        lq   = np.fliplr(lq).copy()
        hq   = np.fliplr(hq).copy()
    if random.random() > 0.5:
        gray = np.flipud(gray).copy()
        lq   = np.flipud(lq).copy()
        hq   = np.flipud(hq).copy()
    return gray, lq, hq


def random_rotate90(gray: np.ndarray, lq: np.ndarray, hq: np.ndarray):
    k = random.choice([0, 1, 2, 3])
    gray = np.rot90(gray, k).copy()
    lq   = np.rot90(lq,   k).copy()
    hq   = np.rot90(hq,   k).copy()
    return gray, lq, hq


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class GuidedFusionDataset(Dataset):
    """
    Loads HQ color images and synthesizes (HQ Gray, LQ RGB, HQ RGB) triplets.

    Args:
        roots      : list of directories containing HQ images
        crop_size  : random crop size (None = no crop, use full image)
        augment    : flip + rotate augmentations
        scale      : LQ downscale factor (2 or 4)
        noise_std  : noise level added to LQ (0–255 scale)
        min_size   : skip images smaller than this in either dimension
    """

    def __init__(
        self,
        roots: list[str],
        crop_size: int = 256,
        augment: bool = True,
        scale: int = 4,
        noise_std: float = 25.0,
        min_size: int = 300,
    ):
        self.crop_size = crop_size
        self.augment   = augment
        self.scale     = scale
        self.noise_std = noise_std
        self.min_size  = min_size

        self.files: list[Path] = []
        for root in roots:
            p = Path(root)
            if not p.exists():
                print(f"[Dataset] WARNING: {root} not found — skipping")
                continue
            for ext in VALID_EXTS:
                self.files.extend(p.rglob(f"*{ext}"))
                self.files.extend(p.rglob(f"*{ext.upper()}"))

        self.files = sorted(set(self.files))
        print(f"[Dataset] Found {len(self.files)} images from {len(roots)} source(s)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]

        # Load HQ color image
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            # Fallback to next image
            return self.__getitem__((idx + 1) % len(self.files))

        hq = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Skip images that are too small
        h, w = hq.shape[:2]
        need = self.crop_size or self.min_size
        if h < need or w < need:
            return self.__getitem__((idx + 1) % len(self.files))

        # Synthesize inputs
        gray = rgb_to_grayscale(hq)          # HxW
        lq   = degrade_rgb(hq, self.scale, self.noise_std)  # HxWx3

        # Augment
        gray_3 = gray[:, :, np.newaxis]      # HxWx1 for joint ops
        if self.augment and self.crop_size:
            gray_3, lq, hq = random_crop(gray_3, lq, hq, self.crop_size)
            gray_3, lq, hq = random_flip(gray_3, lq, hq)
            gray_3, lq, hq = random_rotate90(gray_3, lq, hq)
        elif self.crop_size:
            gray_3 = gray_3[:self.crop_size, :self.crop_size]
            lq     = lq    [:self.crop_size, :self.crop_size]
            hq     = hq    [:self.crop_size, :self.crop_size]

        # HWC → CHW tensors
        gray_t = torch.from_numpy(gray_3.transpose(2, 0, 1)).float()  # [1, H, W]
        lq_t   = torch.from_numpy(lq.transpose(2, 0, 1)).float()      # [3, H, W]
        hq_t   = torch.from_numpy(hq.transpose(2, 0, 1)).float()      # [3, H, W]

        return gray_t, lq_t, hq_t


# ─────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────

def build_dataloaders(
    train_roots: list[str],
    val_roots: list[str],
    crop_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    scale: int = 4,
    noise_std: float = 25.0,
):
    train_ds = GuidedFusionDataset(
        roots=train_roots,
        crop_size=crop_size,
        augment=True,
        scale=scale,
        noise_std=noise_std,
    )
    val_ds = GuidedFusionDataset(
        roots=val_roots,
        crop_size=crop_size,
        augment=False,
        scale=scale,
        noise_std=noise_std,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    print("dataset.py — GuidedFusionDataset ready.")
