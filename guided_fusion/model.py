"""
model.py
========
GuidedFusionNet — Dual-branch encoder + cross-attention fusion + U-Net decoder

Architecture:
  ┌─────────────┐   ┌──────────────┐
  │  LQ RGB     │   │  HQ Gray     │
  │  Encoder    │   │  Encoder     │
  │  (3-ch in)  │   │  (1-ch in)   │
  └──────┬──────┘   └──────┬───────┘
         │                 │
         └────── Fusion ───┘   ← Cross-attention at each scale
                   │
              Bottleneck
                   │
             U-Net Decoder
                   │
            HQ Color Output  [3, H, W]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────────────────────
# Basic building blocks
# ─────────────────────────────────────────────────────────────

class ConvBnReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


# ─────────────────────────────────────────────────────────────
# Channel + Spatial Attention (CBAM-style)
# ─────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        w = self.sigmoid(
            (self.mlp(self.avg(x)) + self.mlp(self.max(x))).view(b, c, 1, 1)
        )
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        w   = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


# ─────────────────────────────────────────────────────────────
# Cross-attention fusion block
# ─────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Fuse color (query) and gray (key/value) features at each scale.
    Gray provides structure; color provides hue. 
    Cross-attention lets color features query where gray has detail.
    """

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.heads   = heads
        self.scale   = (ch // heads) ** -0.5
        self.to_q    = nn.Conv2d(ch, ch, 1, bias=False)
        self.to_k    = nn.Conv2d(ch, ch, 1, bias=False)
        self.to_v    = nn.Conv2d(ch, ch, 1, bias=False)
        self.proj    = nn.Conv2d(ch, ch, 1, bias=False)

        # Channel + spatial attention on concatenated features
        self.ch_attn = ChannelAttention(ch * 2)
        self.sp_attn = SpatialAttention()
        self.fuse    = ConvBnReLU(ch * 2, ch)

    def forward(self, color_feat: torch.Tensor, gray_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = color_feat.shape

        # Cross-attention: color queries gray
        q = rearrange(self.to_q(color_feat), "b (nh d) h w -> b nh (h w) d", nh=self.heads)
        k = rearrange(self.to_k(gray_feat),  "b (nh d) h w -> b nh (h w) d", nh=self.heads)
        v = rearrange(self.to_v(gray_feat),  "b (nh d) h w -> b nh (h w) d", nh=self.heads)

        attn = torch.softmax(torch.einsum("b n i d, b n j d -> b n i j", q, k) * self.scale, dim=-1)
        out  = torch.einsum("b n i j, b n j d -> b n i d", attn, v)
        out  = rearrange(out, "b nh (h w) d -> b (nh d) h w", h=h, w=w)
        out  = self.proj(out)

        # Merge with spatial/channel attention
        cat  = torch.cat([out, gray_feat], dim=1)
        cat  = self.ch_attn(cat)
        cat  = self.sp_attn(cat)
        return self.fuse(cat)


# ─────────────────────────────────────────────────────────────
# Encoder branch
# ─────────────────────────────────────────────────────────────

class EncoderBranch(nn.Module):
    """3-scale encoder for either RGB (3-ch) or Grayscale (1-ch) input."""

    def __init__(self, in_ch: int, base_ch: int = 32):
        super().__init__()
        ch = base_ch
        self.s1 = nn.Sequential(ConvBnReLU(in_ch, ch),     ResBlock(ch))       # full res
        self.s2 = nn.Sequential(ConvBnReLU(ch, ch * 2, stride=2), ResBlock(ch * 2))  # /2
        self.s3 = nn.Sequential(ConvBnReLU(ch * 2, ch * 4, stride=2), ResBlock(ch * 4))  # /4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        return f1, f2, f3   # (full, /2, /4)


# ─────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            ConvBnReLU(in_ch + skip_ch, out_ch),
            ResBlock(out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch from odd dimensions
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ─────────────────────────────────────────────────────────────
# Main Network
# ─────────────────────────────────────────────────────────────

class GuidedFusionNet(nn.Module):
    """
    Input:
        lq_rgb  : [B, 3, H, W]  — low quality color
        hq_gray : [B, 1, H, W]  — high quality grayscale

    Output:
        hq_rgb  : [B, 3, H, W]  — high quality color in [0, 1]
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()
        ch = base_ch
        c2, c4 = ch * 2, ch * 4

        # Dual encoders
        self.color_enc = EncoderBranch(in_ch=3, base_ch=ch)
        self.gray_enc  = EncoderBranch(in_ch=1, base_ch=ch)

        # Cross-attention fusion at each scale
        self.fuse1 = CrossAttentionFusion(ch,  heads=max(1, ch // 8))
        self.fuse2 = CrossAttentionFusion(c2,  heads=max(1, c2 // 16))
        self.fuse3 = CrossAttentionFusion(c4,  heads=max(1, c4 // 16))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBnReLU(c4, c4 * 2, stride=2),
            ResBlock(c4 * 2),
            ResBlock(c4 * 2),
        )

        # Decoder
        self.dec3 = DecoderBlock(in_ch=c4 * 2, skip_ch=c4, out_ch=c4)
        self.dec2 = DecoderBlock(in_ch=c4,      skip_ch=c2, out_ch=c2)
        self.dec1 = DecoderBlock(in_ch=c2,      skip_ch=ch, out_ch=ch)

        # Output head
        self.head = nn.Sequential(
            ConvBnReLU(ch, ch),
            nn.Conv2d(ch, 3, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, lq_rgb: torch.Tensor, hq_gray: torch.Tensor) -> torch.Tensor:
        # Encode both branches
        cf1, cf2, cf3 = self.color_enc(lq_rgb)
        gf1, gf2, gf3 = self.gray_enc(hq_gray)

        # Fuse at each scale (gray guides color)
        f1 = self.fuse1(cf1, gf1)
        f2 = self.fuse2(cf2, gf2)
        f3 = self.fuse3(cf3, gf3)

        # Bottleneck
        b = self.bottleneck(f3)

        # Decode with skip connections from fused features
        d3 = self.dec3(b,  f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)

        return self.head(d1)


# ─────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = GuidedFusionNet(base_ch=32)
    lq  = torch.randn(2, 3, 256, 256)
    hq  = torch.randn(2, 1, 256, 256)
    out = net(lq, hq)
    print(f"Output shape : {out.shape}")       # [2, 3, 256, 256]
    print(f"Parameters   : {count_parameters(net):,}")
