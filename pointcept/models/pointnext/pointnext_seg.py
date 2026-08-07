"""
PointNeXt for Semantic Segmentation in Pointcept.

Reference:
    PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies
    (Qian et al., NeurIPS 2022)
"""

import torch.nn as nn

from pointcept.models.builder import MODELS
from pointcept.models.pointnet2.pointnet2_utils import FeaturePropagation
from .pointnext_utils import StemMLP, EncoderStage


class PointNeXtSeg(nn.Module):
    def __init__(
        self,
        in_channels=6,
        num_classes=13,
        width=32,
        blocks=(0, 0, 0, 0),
        strides=None,
        radii=None,
        nsamples=None,
        fp_channels=None,
        cls_channels=None,
        dropout=0.5,
    ):
        super().__init__()
        strides = strides or [4, 4, 4, 4]
        radii = radii or [0.05, 0.1, 0.2, 0.4]
        nsamples = nsamples or [32, 32, 32, 32]
        blocks = tuple(blocks)
        enc_channels = [width * (2 ** i) for i in range(1, 5)]  # 2C, 4C, 8C, 16C
        fp_channels = fp_channels or [256, 256, 128, 128]
        cls_channels = cls_channels or [128, 128]

        assert len(strides) == len(radii) == len(nsamples) == len(blocks) == 4
        assert len(fp_channels) == 4

        self.stem = StemMLP(in_channels, width)

        in_ch = width
        self.encoder = nn.ModuleList()
        for i, (stride, radius, nsample, out_ch, num_inv) in enumerate(
            zip(strides, radii, nsamples, enc_channels, blocks)
        ):
            self.encoder.append(
                EncoderStage(
                    stride=stride,
                    radius=radius,
                    nsample=nsample,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_inv_blocks=num_inv,
                )
            )
            in_ch = out_ch

        self.fp_layers = nn.ModuleList()
        self.fp_layers.append(
            FeaturePropagation(
                [fp_channels[0]],
                in_channels=enc_channels[3],
                skip_channels=enc_channels[2],
            )
        )
        self.fp_layers.append(
            FeaturePropagation(
                [fp_channels[1]],
                in_channels=fp_channels[0],
                skip_channels=enc_channels[1],
            )
        )
        self.fp_layers.append(
            FeaturePropagation(
                [fp_channels[2]],
                in_channels=fp_channels[1],
                skip_channels=enc_channels[0],
            )
        )
        self.fp_layers.append(
            FeaturePropagation(
                [fp_channels[3], fp_channels[3]],
                in_channels=fp_channels[2],
                skip_channels=0,
            )
        )

        cls_layers = []
        prev_ch = fp_channels[3]
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
        x0 = self.stem(data_dict["feat"])
        o0 = data_dict["offset"].int()

        points = [(p0, x0, o0)]
        for stage in self.encoder:
            points.append(stage(*points[-1]))
        p1, x1, o1 = points[1]
        p2, x2, o2 = points[2]
        p3, x3, o3 = points[3]
        p4, x4, o4 = points[4]

        x = self.fp_layers[0](p3, x3, o3, p4, x4, o4)
        x = self.fp_layers[1](p2, x2, o2, p3, x, o3)
        x = self.fp_layers[2](p1, x1, o1, p2, x, o2)
        x = self.fp_layers[3](p0, None, o0, p1, x, o1)

        return self.cls(x)


@MODELS.register_module("PointNeXt-S")
class PointNeXtS(PointNeXtSeg):
    """PointNeXt-S: C=32, B=(0,0,0,0)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("width", 32)
        kwargs.setdefault("blocks", (0, 0, 0, 0))
        super().__init__(**kwargs)


@MODELS.register_module("PointNeXt-B")
class PointNeXtB(PointNeXtSeg):
    """PointNeXt-B: C=32, B=(1,2,1,1)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("width", 32)
        kwargs.setdefault("blocks", (1, 2, 1, 1))
        super().__init__(**kwargs)


@MODELS.register_module("PointNeXt-L")
class PointNeXtL(PointNeXtSeg):
    """PointNeXt-L: C=32, B=(2,4,2,2)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("width", 32)
        kwargs.setdefault("blocks", (2, 4, 2, 2))
        super().__init__(**kwargs)


@MODELS.register_module("PointNeXt-XL")
class PointNeXtXL(PointNeXtSeg):
    """PointNeXt-XL: C=64, B=(3,6,3,3)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("width", 64)
        kwargs.setdefault("blocks", (3, 6, 3, 3))
        kwargs.setdefault("fp_channels", [512, 512, 256, 256])
        kwargs.setdefault("cls_channels", [256, 256])
        super().__init__(**kwargs)
