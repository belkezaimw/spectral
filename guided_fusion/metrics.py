"""metrics.py — PSNR and SSIM."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10(max_val ** 2 / (mse + 1e-8))


def _gaussian_kernel(size: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g.outer(g)
    g = g / g.sum()
    return g.view(1, 1, size, size).repeat(channels, 1, 1, 1)


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ch = pred.shape[1]
    k  = _gaussian_kernel(window_size, channels=ch).to(pred.device)
    pad = window_size // 2

    mu_p  = F.conv2d(pred,   k, padding=pad, groups=ch)
    mu_t  = F.conv2d(target, k, padding=pad, groups=ch)
    mu_pp = mu_p * mu_p
    mu_tt = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sig_pp = F.conv2d(pred   * pred,   k, padding=pad, groups=ch) - mu_pp
    sig_tt = F.conv2d(target * target, k, padding=pad, groups=ch) - mu_tt
    sig_pt = F.conv2d(pred   * target, k, padding=pad, groups=ch) - mu_pt

    num = (2 * mu_pt + C1) * (2 * sig_pt + C2)
    den = (mu_pp + mu_tt + C1) * (sig_pp + sig_tt + C2)
    return (num / den).mean()
