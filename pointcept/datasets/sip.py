"""
SIP Dataset

Author: Seongyong Kim (syoi0928@gmail.com)
Adapted from the original Pointcept implementation by Xiaoyang Wu (xiaoyang.wu.cs@gmail.com).
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .transform import Compose, TRANSFORMS


@DATASETS.register_module()
class SIPDataset(Dataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "intensity",
        "segment",
    ]

    def __init__(
        self,
        split="train",
        data_root="data/sip",
        transform=None,
        fragmentation=None,
        post_transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        ignore_index=-1,
        loop=1,
    ):
        super(SIPDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.cache = cache
        self.ignore_index = ignore_index
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        self.transform = Compose(transform)
        self.fragmentation = TRANSFORMS.build(fragmentation) if fragmentation is not None else None
        self.post_transform = Compose(post_transform)

        if test_mode:
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} {} set.".format(
                len(self.data_list), self.loop, os.path.basename(self.data_root), split
            )
        )

    def get_data_list(self):
        data_list = []
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)] 
        name = os.path.basename(data_path)
        split = os.path.basename(os.path.dirname(data_path))
       
        if not self.cache:
            data = torch.load(data_path)
        else:
            data_name = data_path.replace(os.path.dirname(self.data_root), "").split(
                "."
            )[0]
            cache_name = "Pointcept" + data_name.replace(os.path.sep, "-")
            data = shared_dict(cache_name)
        
        theta = data.get("theta", np.array([0.0], dtype=np.float32))
        coord = data["coord"]
        color = data.get("rgb", data.get("color"))

        intensity = data.get(
            "intensity",
            np.full((coord.shape[0],1), -1, dtype=coord.dtype),
            )

        normal = data.get(
            "normal",
            np.full((coord.shape[0], 3), -1, dtype=coord.dtype),
            )

        segment = data.get(
            "semantic_gt",
            np.full(coord.shape[0], -1, dtype=np.int32),
            ).reshape(-1)
        # intensity = np.full(coord.shape[0], -1, dtype=coord.dtype)
        
        data_dict = dict(
            name=name,
            split=split,
            theta=theta,
            coord=coord,
            color=color,
            intensity=intensity,
            normal=normal,
            segment=segment,
            # instance=instance,
        )

        data_dict["index_valid_keys"] = list(self.VALID_ASSETS)
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)])

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(segment=data_dict["segment"].copy())

        fragment_list = self.fragmentation(data_dict)
        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        
        result_dict["fragment_list"] = fragment_list
        return result_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(
                name=data_dict["name"],
                segment=data_dict["segment"].copy(),
                )
        # result_dict = dict(segment=data_dict.pop("segment"), name=data_dict.pop("name"))
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")
            result_dict["origin_coord"] = data_dict.pop("origin_coord")

        data_dict_list = []
        if hasattr(self, "aug_transform"):
            for aug in self.aug_transform:
                data_dict_list.append(aug(deepcopy(data_dict)))
        else:
            data_dict_list.append(data_dict)

        fragment_list = []
        
        for data in data_dict_list:
            parts = self.fragmentation(data) if self.fragmentation is not None else [data]
            for p in parts:
                fragment_list.append(self.post_transform(p))

            # if self.test_voxelize is not None:
            #     data_part_list = self.test_voxelize(data)
            # else:
            #     data["index"] = np.arange(data["coord"].shape[0])
            #     data_part_list = [data]
            # for data_part in data_part_list:
            #     if self.test_crop:
            #         data_part = self.test_crop(data_part)
            #     else:
            #         data_part = [data_part]
            #     fragment_list += data_part

        # for i in range(len(fragment_list)):
        #     fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        return result_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
