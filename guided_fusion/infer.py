"""
infer.py
========
Run inference on a real image pair: LQ RGB + HQ Grayscale → HQ Color

Usage:
    uv run gf-infer \
        --checkpoint checkpoints/best_model.pth \
        --lq_rgb     path/to/low_quality_color.jpg \
        --hq_gray    path/to/high_quality_bw.jpg   \
        --output     result.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from guided_fusion.model import GuidedFusionNet


def load_image_rgb(path: str) -> np.ndarray:
    """Load image as float32 [0,1] RGB."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def load_image_gray(path: str) -> np.ndarray:
    """Load image as float32 [0,1] grayscale HxW."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img.astype(np.float32) / 255.0


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    """HxW or HxWxC numpy → 1xCxHxW tensor."""
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)


def from_tensor(t: torch.Tensor) -> np.ndarray:
    """1x3xHxW tensor → HxWx3 uint8."""
    arr = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return (arr * 255).astype(np.uint8)


def tiled_infer(
    model: torch.nn.Module,
    lq: torch.Tensor,
    gray: torch.Tensor,
    tile: int = 512,
    overlap: int = 32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Process large images in overlapping tiles to avoid OOM.
    lq   : [1, 3, H, W]
    gray : [1, 1, H, W]
    """
    _, _, H, W = lq.shape
    out = torch.zeros(1, 3, H, W, device=device)
    cnt = torch.zeros(1, 1, H, W, device=device)

    step = tile - overlap
    ys   = list(range(0, H - tile, step)) + [max(0, H - tile)]
    xs   = list(range(0, W - tile, step)) + [max(0, W - tile)]

    model.eval()
    with torch.no_grad():
        for y in ys:
            for x in xs:
                y2 = min(y + tile, H)
                x2 = min(x + tile, W)
                lq_t   = lq  [:, :, y:y2, x:x2].to(device)
                gray_t = gray[:, :, y:y2, x:x2].to(device)
                pred   = model(lq_t, gray_t)
                out[:, :, y:y2, x:x2] += pred.cpu()
                cnt[:,  :, y:y2, x:x2] += 1.0

    return out / cnt.clamp(min=1)


def main():
    parser = argparse.ArgumentParser(description="GuidedFusionNet Inference")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--lq_rgb",     required=True, help="Low-quality color image")
    parser.add_argument("--hq_gray",    required=True, help="High-quality B&W image")
    parser.add_argument("--output",     default="result.png", help="Output path")
    parser.add_argument("--tile_size",  type=int, default=512)
    parser.add_argument("--base_ch",    type=int, default=32)
    parser.add_argument("--no_tile",    action="store_true", help="Full image (needs enough VRAM)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Infer] Device: {device}")

    # Load model
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model = GuidedFusionNet(base_ch=args.base_ch)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    print(f"[Infer] Model loaded from epoch {ckpt.get('epoch', '?')}")

    # Load inputs
    lq_rgb  = load_image_rgb(args.lq_rgb)
    hq_gray = load_image_gray(args.hq_gray)

    H, W = hq_gray.shape[:2]
    print(f"[Infer] Input size: {W}×{H}")

    # Resize LQ to match HQ if needed
    lq_h, lq_w = lq_rgb.shape[:2]
    if (lq_h, lq_w) != (H, W):
        print(f"[Infer] Resizing LQ {lq_w}×{lq_h} → {W}×{H}")
        lq_rgb = cv2.resize(lq_rgb, (W, H), interpolation=cv2.INTER_CUBIC)

    lq_t   = to_tensor(lq_rgb)       # [1, 3, H, W]
    gray_t = to_tensor(hq_gray)      # [1, 1, H, W]

    # Infer
    if args.no_tile:
        with torch.no_grad():
            pred = model(lq_t.to(device), gray_t.to(device)).cpu()
    else:
        pred = tiled_infer(model, lq_t, gray_t, tile=args.tile_size, device=device)

    # Save
    result_rgb = from_tensor(pred)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, result_bgr)
    print(f"[Infer] Saved → {args.output}")

    # Side-by-side comparison
    comp_path = Path(args.output).with_stem(Path(args.output).stem + "_compare")
    lq_u8     = (lq_rgb * 255).astype(np.uint8)
    gray_u8   = cv2.cvtColor((hq_gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    comp      = np.concatenate([
        cv2.cvtColor(lq_u8, cv2.COLOR_RGB2BGR),
        gray_u8,
        result_bgr,
    ], axis=1)
    cv2.imwrite(str(comp_path), comp)
    print(f"[Infer] Comparison (LQ | Gray | Result) → {comp_path}")


if __name__ == "__main__":
    main()
