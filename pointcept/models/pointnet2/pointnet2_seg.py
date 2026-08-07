"""
PointNet++ (SSG) for Semantic Segmentation in Pointcept.

Reference:
    PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
    (Qi et al., NeurIPS 2017)

Author: Pointcept-SIP
"""

import torch.nn as nn

from pointcept.models.builder import MODELS
from .pointnet2_utils import SetAbstraction, FeaturePropagation


class PointNet2Seg(nn.Module):
    def __init__(
        self,
        in_channels=6,
        num_classes=13,
        strides=None,
        radii=None,
        nsamples=None,
        sa_mlps=None,
        fp_mlps=None,
        cls_channels=None,
        dropout=0.5,
    ):
        super().__init__()
        strides = strides or [4, 4, 4, 4]
        radii = radii or [0.05, 0.1, 0.2, 0.4]
        nsamples = nsamples or [32, 32, 32, 32]
        sa_mlps = sa_mlps or [
            [32, 32, 64],
            [64, 64, 128],
            [128, 128, 256],
            [256, 256, 512],
        ]
        fp_mlps = fp_mlps or [
            [256, 256],
            [256, 256],
            [256, 128],
            [128, 128, 128],
        ]
        cls_channels = cls_channels or [128, 128]

        assert len(strides) == len(radii) == len(nsamples) == len(sa_mlps) == 4
        assert len(fp_mlps) == 4

        sa_out_channels = [mlp[-1] for mlp in sa_mlps]
        fp_out_channels = [mlp[-1] for mlp in fp_mlps]

        in_ch = in_channels
        self.sa_layers = nn.ModuleList()
        for stride, radius, nsample, mlp in zip(strides, radii, nsamples, sa_mlps):
            self.sa_layers.append(
                SetAbstraction(stride, radius, nsample, mlp, in_ch)
            )
            in_ch = mlp[-1]

        self.fp_layers = nn.ModuleList()
        # fp4: interp(sa4) + skip(sa3)
        self.fp_layers.append(
            FeaturePropagation(
                fp_mlps[0],
                in_channels=sa_out_channels[3],
                skip_channels=sa_out_channels[2],
            )
        )
        # fp3: interp(fp4) + skip(sa2)
        self.fp_layers.append(
            FeaturePropagation(
                fp_mlps[1],
                in_channels=fp_out_channels[0],
                skip_channels=sa_out_channels[1],
            )
        )
        # fp2: interp(fp3) + skip(sa1)
        self.fp_layers.append(
            FeaturePropagation(
                fp_mlps[2],
                in_channels=fp_out_channels[1],
                skip_channels=sa_out_channels[0],
            )
        )
        # fp1: interp(fp2), no skip from raw input features
        self.fp_layers.append(
            FeaturePropagation(
                fp_mlps[3],
                in_channels=fp_out_channels[2],
                skip_channels=0,
            )
        )

        cls_layers = []
        prev_ch = fp_out_channels[3]
        for ch in cls_channels:
            cls_layers.extend(
                [
                    nn.Linear(prev_ch, ch, bias=False),
                    nn.BatchNorm1d(ch),
                    nn.ReLU(inplace=True),
                ]
            )
            prev_ch = ch
        if dropout > 0:
            cls_layers.append(nn.Dropout(dropout))
        cls_layers.append(nn.Linear(prev_ch, num_classes))
        self.cls = nn.Sequential(*cls_layers)

    def forward(self, data_dict):
        p0 = data_dict["coord"]
        x0 = data_dict["feat"]
        o0 = data_dict["offset"].int()

        points = [(p0, x0, o0)]
        for sa in self.sa_layers:
            points.append(sa(*points[-1]))
        p1, x1, o1 = points[1]
        p2, x2, o2 = points[2]
        p3, x3, o3 = points[3]
        p4, x4, o4 = points[4]

        x = self.fp_layers[0](p3, x3, o3, p4, x4, o4)
        x = self.fp_layers[1](p2, x2, o2, p3, x, o3)
        x = self.fp_layers[2](p1, x1, o1, p2, x, o2)
        x = self.fp_layers[3](p0, None, o0, p1, x, o1)

        return self.cls(x)


@MODELS.register_module("PointNet2-SSG")
class PointNet2SSG(PointNet2Seg):
    """Standard PointNet++ SSG segmentation backbone."""


@MODELS.register_module("PointNet2-SSG-Lite")
class PointNet2SSGLite(PointNet2Seg):
    """Lightweight PointNet++ SSG for faster baseline experiments."""

    def __init__(self, **kwargs):
        kwargs.setdefault(
            "sa_mlps",
            [[16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128, 256]],
        )
        kwargs.setdefault("fp_mlps", [[256, 128], [128, 128], [128, 64], [64, 64, 64]])
        kwargs.setdefault("cls_channels", [64])
        kwargs.setdefault("strides", [4, 4, 4, 4])
        kwargs.setdefault("radii", [0.05, 0.1, 0.2, 0.4])
        kwargs.setdefault("nsamples", [16, 16, 16, 16])
        super().__init__(**kwargs)
