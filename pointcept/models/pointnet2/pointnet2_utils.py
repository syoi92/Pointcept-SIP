"""
PointNet++ building blocks for Pointcept.

Adapted from the PointNet++ SSG semantic segmentation architecture to use
batched point clouds with cumulative offsets (coord / feat / offset).
"""

import torch
import torch.nn as nn
import pointops


def compute_new_offset(offset, ratio):
    """Compute cumulative offsets after FPS downsampling by *ratio*."""
    if ratio <= 1:
        return offset.int()
    device = offset.device
    new_offset = []
    count = 0
    prev = 0
    for i in range(offset.shape[0]):
        end = offset[i].item()
        n = max(1, (end - prev) // ratio)
        count += n
        new_offset.append(count)
        prev = end
    return torch.tensor(new_offset, dtype=torch.int32, device=device)


class SharedMLP2d(nn.Module):
    """Shared MLP on grouped points using 1x1 Conv2d, followed by max pooling."""

    def __init__(self, channels, bn=True):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=1, bias=False)
            )
            if bn:
                layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x: (m, nsample, c)
        x = x.permute(2, 0, 1).unsqueeze(0)  # (1, c, m, nsample)
        x = self.layers(x)
        x = x.max(dim=-1)[0].squeeze(0).permute(1, 0)  # (m, c_out)
        return x


class SharedMLP1d(nn.Module):
    """Shared MLP on per-point features."""

    def __init__(self, channels, bn=True):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Linear(channels[i], channels[i + 1], bias=False))
            if bn:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SetAbstraction(nn.Module):
    """PointNet++ set abstraction: FPS, ball query, shared MLP, max pool."""

    def __init__(self, stride, radius, nsample, mlp, in_channels):
        super().__init__()
        self.stride = stride
        self.radius = radius
        self.nsample = nsample
        self.mlp = SharedMLP2d([in_channels + 3] + list(mlp))

    def forward(self, p, x, o):
        p = p.contiguous()
        x = x.contiguous()
        if self.stride > 1:
            new_o = compute_new_offset(o, self.stride)
            idx = pointops.farthest_point_sampling(p, o.int(), new_o)
            new_p = p[idx.long(), :].contiguous()
        else:
            new_p, new_o = p, o.int()

        grouped, _ = pointops.ball_query_and_group(
            x,
            p,
            offset=o.int(),
            new_xyz=new_p,
            new_offset=new_o,
            nsample=self.nsample,
            max_radio=self.radius,
            min_radio=0.0,
            with_xyz=True,
        )
        new_x = self.mlp(grouped)
        return new_p, new_x, new_o


class FeaturePropagation(nn.Module):
    """PointNet++ feature propagation with distance-weighted interpolation."""

    def __init__(self, mlp, in_channels, skip_channels=0):
        super().__init__()
        self.mlp = SharedMLP1d([in_channels + skip_channels] + list(mlp))

    def forward(self, p_fine, x_fine, o_fine, p_coarse, x_coarse, o_coarse):
        p_fine = p_fine.contiguous()
        if p_coarse is None:
            new_x = x_fine.contiguous()
        else:
            new_x = pointops.interpolation(
                p_coarse.contiguous(),
                p_fine,
                x_coarse.contiguous(),
                o_coarse.int(),
                o_fine.int(),
            )
        if x_fine is not None:
            new_x = torch.cat([new_x, x_fine.contiguous()], dim=1)
        return self.mlp(new_x)
