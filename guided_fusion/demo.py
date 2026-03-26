"""
demo.py
=======
Quick demo that works WITHOUT any dataset.
Downloads 5 sample images from the internet and runs inference.

Usage:
    uv run gf-demo
    uv run gf-demo --checkpoint checkpoints/best_model.pth  (if trained)
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

from guided_fusion.dataset import degrade_rgb, rgb_to_grayscale
from guided_fusion.model import GuidedFusionNet


# Free nature/forest sample images (Unsplash, license-free)
SAMPLE_URLS = [
    "https://images.unsplash.com/photo-1448375240586-882707db888b?w=800&q=80",  # forest
    "https://images.unsplash.com/photo-1518173946687-a4c8892bbd9f?w=800&q=80",  # trees
    "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800&q=80",  # forest path
]


def download_sample(url: str, path: Path):
    if path.exists():
        return
    print(f"  Downloading sample image...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(path, "wb") as f:
        f.write(r.read())


def run_demo(checkpoint: str | None, out_dir: Path, base_ch: int = 32):
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Demo] Device: {device}")

    # Load or init model
    model = GuidedFusionNet(base_ch=base_ch).to(device)
    if checkpoint and Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"[Demo] Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
    else:
        print("[Demo] No checkpoint — using untrained model (results will be poor)")
        print("[Demo] Train first with: uv run gf-train ...")
    model.eval()

    for i, url in enumerate(SAMPLE_URLS):
        img_path = sample_dir / f"sample_{i+1}.jpg"
        try:
            download_sample(url, img_path)
        except Exception as e:
            print(f"  [skip] Could not download sample {i+1}: {e}")
            # Create a synthetic test image instead
            dummy = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
            cv2.imwrite(str(img_path), dummy)

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        # Resize to a safe size for inference
        # On CPU: 256×256 max (attention is O(H²W²) — too slow/large otherwise)
        # On GPU: can go up to 512×512
        max_side = 512 if device.type == "cuda" else 256
        h0, w0 = bgr.shape[:2]
        scale_f = min(max_side / h0, max_side / w0, 1.0)
        if scale_f < 1.0:
            new_w = int(w0 * scale_f)
            new_h = int(h0 * scale_f)
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        hq = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Synthesize inputs
        gray = rgb_to_grayscale(hq)
        lq   = degrade_rgb(hq, scale=4, noise_std=25.0)

        # To tensors
        gray_t = torch.from_numpy(gray[np.newaxis, np.newaxis]).float().to(device)
        lq_t   = torch.from_numpy(lq.transpose(2, 0, 1)[np.newaxis]).float().to(device)

        with torch.no_grad():
            pred = model(lq_t, gray_t).squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()

        # Save side-by-side: LQ | HQ Gray | Prediction | GT
        lq_u8   = (lq   * 255).astype(np.uint8)
        gray_u8 = (np.stack([gray, gray, gray], axis=-1) * 255).astype(np.uint8)
        pred_u8 = (pred * 255).astype(np.uint8)
        gt_u8   = (hq   * 255).astype(np.uint8)

        comp = np.concatenate([
            cv2.cvtColor(lq_u8,   cv2.COLOR_RGB2BGR),
            cv2.cvtColor(gray_u8, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(pred_u8, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(gt_u8,   cv2.COLOR_RGB2BGR),
        ], axis=1)

        # Add labels
        h = comp.shape[0]
        labels = ["LQ RGB (input)", "HQ Gray (input)", "Prediction", "Ground Truth"]
        w_each = comp.shape[1] // 4
        for j, label in enumerate(labels):
            cv2.putText(comp, label, (j * w_each + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        out_path = out_dir / f"demo_{i+1}.jpg"
        cv2.imwrite(str(out_path), comp)
        print(f"[Demo] Saved: {out_path}  (LQ | HQ Gray | Pred | GT)")

    print(f"\n[Demo] Done. Results in: {out_dir}/")
    print("[Demo] Each image shows:  LQ RGB | HQ Grayscale | Prediction | Ground Truth")


def main():
    parser = argparse.ArgumentParser(description="Quick demo")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out_dir",    default="./demo_output")
    parser.add_argument("--base_ch",    type=int, default=32)
    args = parser.parse_args()
    run_demo(args.checkpoint, Path(args.out_dir), args.base_ch)


if __name__ == "__main__":
    main()