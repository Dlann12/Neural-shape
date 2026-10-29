"""Multi-view 3D occupancy reconstruction network (NeuralShape).

Pipeline:
  1. ImageEncoder2D     - small U-Net producing 32x32 feature maps per view.
  2. ViewFeatureAggregator - unprojects features + raw RGB into the voxel grid with the calibrated camera
                          rig, adds pose/depth encoding, and fuses an arbitrary number of views with
                          mean / std (photo-consistency) / max / attention pooling.
  3. ShapeDecoder3D     - 3D U-Net (G -> G/2 -> G) predicting per-voxel occupancy logits.
"""
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.camera import NUM_VIEWS, pose_vector, project_points, resolve_view_indices, voxel_centers


@dataclass
class ModelConfig:
    feat_channels: int = 32
    hidden_3d: int = 32
    grid_size: int = 24

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ModelConfig":
        data = data or {}
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


def _gn(channels: int) -> nn.GroupNorm:
    """GroupNorm keeps statistics independent of batch size and number of views."""
    groups = 8 if channels % 8 == 0 else (4 if channels % 4 == 0 else 1)
    return nn.GroupNorm(groups, channels)


def _conv2d(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False), _gn(cout), nn.ReLU(inplace=True))


def _conv3d(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False), _gn(cout), nn.ReLU(inplace=True))


class ImageEncoder2D(nn.Module):
    """(N, 3, H, W) -> (N, C, H/2, W/2) with a coarse-to-fine skip connection."""

    def __init__(self, out_channels: int = 32):
        super().__init__()
        self.stem = _conv2d(3, 16)
        self.down1 = nn.Sequential(_conv2d(16, 32, stride=2), _conv2d(32, 32))
        self.down2 = nn.Sequential(_conv2d(32, 64, stride=2), _conv2d(64, 64))
        self.fuse = nn.Sequential(_conv2d(64 + 32, out_channels), nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.down1(self.stem(x))
        f2 = self.down2(f1)
        f2_up = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([f1, f2_up], dim=1))


class ViewFeatureAggregator(nn.Module):
    """Unprojects per-view features into the voxel grid and fuses them across views."""

    POSE_DIM = 4

    def __init__(self, feat_channels: int = 32, grid_size: int = 24):
        super().__init__()
        self.feat_channels = feat_channels
        self.grid_size = grid_size

        coords = voxel_centers(grid_size)
        uv, depth = [], []
        for v in range(NUM_VIEWS):
            u_n, v_n, d = project_points(coords, v)
            uv.append(torch.from_numpy(np.stack([u_n, v_n], axis=-1)).view(1, -1, 2))
            depth.append(torch.from_numpy(d).view(1, -1))
        self.register_buffer("proj_uv", torch.stack(uv, dim=0), persistent=False)       # (V, 1, G^3, 2)
        self.register_buffer("view_depth", torch.stack(depth, dim=0), persistent=False)  # (V, 1, G^3)
        poses = torch.stack([torch.from_numpy(pose_vector(v)) for v in range(NUM_VIEWS)], dim=0)
        self.register_buffer("poses", poses, persistent=False)                           # (V, 4)

        in_ch = feat_channels + 3 + 1 + self.POSE_DIM
        self.per_view = nn.Sequential(
            nn.Conv1d(in_ch, feat_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(feat_channels, feat_channels, 1),
            nn.ReLU(inplace=True),
        )
        self.attn_head = nn.Conv1d(feat_channels, 1, 1)

    @property
    def out_channels(self) -> int:
        # attention + mean + std + max over features, plus mean & std of raw RGB
        return 4 * self.feat_channels + 6

    def forward(
        self,
        feat_2d: torch.Tensor,
        images: torch.Tensor,
        B: int,
        V: int,
        view_indices: Sequence[int],
    ) -> torch.Tensor:
        """
        feat_2d: (B*V, C, Hf, Wf), images: (B*V, 3, H, W)
        Returns the fused volume (B, out_channels, G, G, G).
        """
        G, C = self.grid_size, self.feat_channels
        N = G ** 3
        vi = torch.as_tensor(list(view_indices), device=feat_2d.device, dtype=torch.long)

        uv = self.proj_uv[vi].unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * V, 1, N, 2)
        feat = F.grid_sample(feat_2d, uv, mode="bilinear", padding_mode="border", align_corners=True).view(B * V, C, N)
        rgb = F.grid_sample(images, uv, mode="bilinear", padding_mode="border", align_corners=True).view(B * V, 3, N)

        depth = self.view_depth[vi].unsqueeze(0).expand(B, -1, -1, -1).reshape(B * V, 1, N)
        pose = self.poses[vi].unsqueeze(0).expand(B, -1, -1).reshape(B * V, self.POSE_DIM, 1).expand(-1, -1, N)

        h = self.per_view(torch.cat([feat, rgb, depth, pose], dim=1)).view(B, V, C, N)
        attn = torch.softmax(self.attn_head(h.view(B * V, C, N)).view(B, V, 1, N), dim=1)

        rgb = rgb.view(B, V, 3, N)
        fused = torch.cat([
            (h * attn).sum(dim=1),
            h.mean(dim=1),
            _std(h),
            h.max(dim=1).values,
            rgb.mean(dim=1),
            _std(rgb),
        ], dim=1)
        return fused.view(B, self.out_channels, G, G, G)


def _std(x: torch.Tensor) -> torch.Tensor:
    """Std over the view axis (dim=1); zero for a single view. Cross-view disagreement is the
    photo-consistency cue that separates real surfaces from empty space inside the visual hull."""
    if x.shape[1] == 1:
        return torch.zeros_like(x[:, 0])
    mean = x.mean(dim=1)
    var = (x * x).mean(dim=1) - mean * mean
    return torch.sqrt(var.clamp_min(0.0) + 1e-6)


class ResBlock3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1, bias=False), _gn(ch), nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1, bias=False), _gn(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.body(x), inplace=True)


class ShapeDecoder3D(nn.Module):
    """3D U-Net: full-res encoder -> half-res context blocks -> upsample + skip -> occupancy logits."""

    def __init__(self, in_channels: int, hidden_channels: int = 32):
        super().__init__()
        h = hidden_channels
        self.reduce = nn.Sequential(nn.Conv3d(in_channels, h, 1, bias=False), _gn(h), nn.ReLU(inplace=True))
        self.enc = ResBlock3D(h)
        self.down = _conv3d(h, 2 * h, stride=2)
        self.mid = nn.Sequential(ResBlock3D(2 * h), ResBlock3D(2 * h))
        self.up = nn.Sequential(nn.Conv3d(2 * h, h, 1, bias=False), _gn(h), nn.ReLU(inplace=True))
        self.dec = nn.Sequential(_conv3d(2 * h, h), ResBlock3D(h))
        self.head = nn.Conv3d(h, 1, 1)
        nn.init.constant_(self.head.bias, -1.5)  # sigmoid(-1.5) ~= 0.18, close to the mean occupancy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.enc(self.reduce(x))
        mid = self.mid(self.down(skip))
        up = self.up(F.interpolate(mid, size=skip.shape[-3:], mode="trilinear", align_corners=False))
        return self.head(self.dec(torch.cat([skip, up], dim=1))).squeeze(1)


class MultiViewOccupancyNet(nn.Module):
    """End-to-end multi-view 3D shape reconstruction model; accepts any subset of the 8 calibrated views."""

    def __init__(self, feat_channels: int = 32, hidden_3d: int = 32, grid_size: int = 24):
        super().__init__()
        self.config = ModelConfig(feat_channels=feat_channels, hidden_3d=hidden_3d, grid_size=grid_size)
        self.grid_size = grid_size
        self.encoder = ImageEncoder2D(out_channels=feat_channels)
        self.aggregator = ViewFeatureAggregator(feat_channels=feat_channels, grid_size=grid_size)
        self.decoder = ShapeDecoder3D(in_channels=self.aggregator.out_channels, hidden_channels=hidden_3d)

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "MultiViewOccupancyNet":
        return cls(feat_channels=cfg.feat_channels, hidden_3d=cfg.hidden_3d, grid_size=cfg.grid_size)

    def forward(
        self,
        images: torch.Tensor,
        view_indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        images: (B, V, 3, H, W); view_indices[i] is the calibrated camera of image slot i.
        Returns (logits, probabilities), each (B, G, G, G).
        """
        B, V, C, H, W = images.shape
        view_indices = resolve_view_indices(V, view_indices)
        flat = images.reshape(B * V, C, H, W)
        vol = self.aggregator(self.encoder(flat), flat, B=B, V=V, view_indices=view_indices)
        logits = self.decoder(vol)
        return logits, torch.sigmoid(logits)


def occupancy_loss(logits: torch.Tensor, targets: torch.Tensor, bce_weight: float = 0.65, iou_weight: float = 0.35) -> torch.Tensor:
    """Binary cross-entropy + differentiable soft-IoU loss (PRD Section 6)."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    union = torch.sum(probs + targets - probs * targets, dim=dims) + 1e-6
    soft_iou_loss = 1.0 - torch.mean(intersection / union)
    return bce_weight * bce + iou_weight * soft_iou_loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
