"""
losses.py
=========
Combined loss for guided image restoration.

  L = w_char * Charbonnier(pred, gt)
    + w_perc * Perceptual(pred, gt)   [VGG feature matching]
    + w_grad * Gradient(pred, gt)     [edge sharpness]
    + w_ssim * (1 - SSIM(pred, gt))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Charbonnier (robust L1)
# ─────────────────────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


# ─────────────────────────────────────────────────────────────
# Gradient loss  (encourages sharp edges)
# ─────────────────────────────────────────────────────────────

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel kernels
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer("ky", ky.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gx_p = F.conv2d(pred,   self.kx, padding=1, groups=3)
        gy_p = F.conv2d(pred,   self.ky, padding=1, groups=3)
        gx_t = F.conv2d(target, self.kx, padding=1, groups=3)
        gy_t = F.conv2d(target, self.ky, padding=1, groups=3)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


# ─────────────────────────────────────────────────────────────
# SSIM loss
# ─────────────────────────────────────────────────────────────

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g.outer(g)
    return g / g.sum()


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11):
        super().__init__()
        k = _gaussian_kernel(window_size)
        self.register_buffer("kernel", k.view(1, 1, window_size, window_size).repeat(3, 1, 1, 1))
        self.pad = window_size // 2
        self.C1  = 0.01 ** 2
        self.C2  = 0.03 ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu_p  = F.conv2d(pred,   self.kernel, padding=self.pad, groups=3)
        mu_t  = F.conv2d(target, self.kernel, padding=self.pad, groups=3)
        mu_pp = mu_p * mu_p
        mu_tt = mu_t * mu_t
        mu_pt = mu_p * mu_t

        sig_pp = F.conv2d(pred   * pred,   self.kernel, padding=self.pad, groups=3) - mu_pp
        sig_tt = F.conv2d(target * target, self.kernel, padding=self.pad, groups=3) - mu_tt
        sig_pt = F.conv2d(pred   * target, self.kernel, padding=self.pad, groups=3) - mu_pt

        ssim = ((2 * mu_pt + self.C1) * (2 * sig_pt + self.C2)) / \
               ((mu_pp + mu_tt + self.C1) * (sig_pp + sig_tt + self.C2))
        return 1.0 - ssim.mean()


# ─────────────────────────────────────────────────────────────
# Combined loss
# ─────────────────────────────────────────────────────────────

class GuidedFusionLoss(nn.Module):
    def __init__(
        self,
        w_char: float = 0.6,
        w_grad: float = 0.2,
        w_ssim: float = 0.2,
    ):
        super().__init__()
        self.char = CharbonnierLoss()
        self.grad = GradientLoss()
        self.ssim = SSIMLoss()
        self.w_char = w_char
        self.w_grad = w_grad
        self.w_ssim = w_ssim

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        l_char = self.char(pred, target)
        l_grad = self.grad(pred, target)
        l_ssim = self.ssim(pred, target)
        total  = self.w_char * l_char + self.w_grad * l_grad + self.w_ssim * l_ssim
        return {
            "total": total,
            "char":  l_char,
            "grad":  l_grad,
            "ssim":  l_ssim,
        }
