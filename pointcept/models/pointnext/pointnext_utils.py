"""
PointNeXt building blocks for Pointcept.

Reference:
    PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies
    (Qian et al., NeurIPS 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pointops

from pointcept.models.pointnet2.pointnet2_utils import (
    compute_new_offset,
    SharedMLP1d,
    SharedMLP2d,
    FeaturePropagation,
)


def normalize_grouped_xyz(grouped, radius):
    """Relative position normalization (divide offsets by query radius)."""
    grouped = grouped.clone()
    grouped[:, :, :3] = grouped[:, :, :3] / radius
    return grouped


def ball_query_group(
    x,
    p,
    o,
    radius,
    nsample,
    new_p=None,
    new_o=None,
    normalize_xyz=True,
):
    if new_p is None:
        new_p = p
    if new_o is None:
        new_o = o
    grouped, _ = pointops.ball_query_and_group(
        x.contiguous(),
        p.contiguous(),
        offset=o.int(),
        new_xyz=new_p.contiguous(),
        new_offset=new_o.int(),
        nsample=nsample,
        max_radio=radius,
        min_radio=0.0,
        with_xyz=True,
    )
    if normalize_xyz:
        grouped = normalize_grouped_xyz(grouped, radius)
    return grouped


class StemMLP(nn.Module):
    """Stem MLP mapping input features to width C."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = SharedMLP1d([in_channels, out_channels, out_channels])

    def forward(self, x):
        return self.mlp(x)


class SetAbstractionNeXt(nn.Module):
    """PointNeXt set abstraction with rel-pos normalization."""

    def __init__(
        self,
        stride,
        radius,
        nsample,
        in_channels,
        out_channels,
        num_layers=2,
    ):
        super().__init__()
        self.stride = stride
        self.radius = radius
        self.nsample = nsample

        if num_layers <= 1:
            mlp_channels = [in_channels + 3, out_channels]
        else:
            mid_channels = out_channels // 2 if stride > 1 else out_channels
            mlp_channels = [in_channels + 3, mid_channels, out_channels]

        self.mlp = SharedMLP2d(mlp_channels)

    def forward(self, p, x, o):
        p = p.contiguous()
        x = x.contiguous()
        if self.stride > 1:
            new_o = compute_new_offset(o, self.stride)
            idx = pointops.farthest_point_sampling(p, o.int(), new_o)
            new_p = p[idx.long(), :].contiguous()
        else:
            new_p, new_o = p, o.int()

        grouped = ball_query_group(
            x,
            p,
            o,
            self.radius,
            self.nsample,
            new_p=new_p,
            new_o=new_o,
        )
        new_x = self.mlp(grouped)
        return new_p, new_x, new_o


class InvResMLP(nn.Module):
    """Inverted residual MLP block at fixed resolution."""

    def __init__(
        self,
        channels,
        radius,
        nsample,
        expansion=4,
    ):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.local = SharedMLP2d([channels + 3, channels])
        hidden = channels * expansion
        self.pwconv = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, p, x, o):
        identity = x
        grouped = ball_query_group(
            x,
            p,
            o,
            self.radius,
            self.nsample,
            new_p=p,
            new_o=o,
        )
        x = self.local(grouped)
        x = self.pwconv(x)
        x = F.relu(x + identity)
        return p, x, o


class EncoderStage(nn.Module):
    """One encoder stage: SA (+ optional InvResMLP blocks)."""

    def __init__(
        self,
        stride,
        radius,
        nsample,
        in_channels,
        out_channels,
        num_inv_blocks=0,
    ):
        super().__init__()
        sa_layers = 1 if num_inv_blocks > 0 else 2
        self.sa = SetAbstractionNeXt(
            stride=stride,
            radius=radius,
            nsample=nsample,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=sa_layers,
        )
        self.inv_blocks = nn.ModuleList(
            [
                InvResMLP(out_channels, radius, nsample)
                for _ in range(num_inv_blocks)
            ]
        )

    def forward(self, p, x, o):
        p, x, o = self.sa(p, x, o)
        for block in self.inv_blocks:
            p, x, o = block(p, x, o)
        return p, x, o
