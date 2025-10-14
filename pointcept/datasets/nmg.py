"""
NMG Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
from .defaults import DefaultDataset
from .builder import DATASETS
import glob
import laspy
import numpy as np

import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict

from .builder import DATASETS, build_dataset
from .transform import Compose, TRANSFORMS



@DATASETS.register_module()
class NMGDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "strength",
        "segment",
    ]

    def __init__(self, ignore_index=-1, **kwargs):
        self.ignore_index = ignore_index
        super().__init__(ignore_index=ignore_index, **kwargs)
        self.ann_folder = ann_folder = '/nas2/YJ/DATA/NMG/2D_RAW/MSD/cropped1024/{}/annfiles/'.format(self.split)

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.las"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.las"))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        
        # 3D 
        las = laspy.read(data_path)
        #coord = np.vstack((las.x - 512.0 , las.y - 512.0, las.z - las.z.min())).transpose()
        coord = np.vstack((las.x, las.y, las.z - las.z.min())).transpose()
        segment = las.classification - 1
        strength = (las.intensity - las.intensity.min()) / (las.intensity.max() - las.intensity.min()) # scale strength to [0, 1]
        strength = strength.reshape([-1, 1])

        data_dict["name"] = name
        data_dict["coord"] = coord.astype(np.float32)
        data_dict["segment"] = segment.astype(np.int32)
        data_dict["strength"] = strength.astype(np.float32)

        # 2D
        target_file = os.path.join(self.ann_folder, name.replace('las', 'txt')) ## need to check
        with open(target_file, 'r') as f :
            lines = f.readlines()

        boxes = []
        classes = []
        for line in lines :
            anno = line.strip().split(" ")
            obbox = anno[:8]
            obbox = [float(f) for f in obbox]
            boxes.append(obbox)
            label = anno[8]
            if label == 'pylon' :
                classes.append(0) # 0 for pylon 
            else :
                classes.append(1) # 1 for span(powerline)

        w, h = 1024, 1024 ## need to be args
        las_id = torch.tensor([idx])

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 8)
        classes = torch.tensor(classes, dtype=torch.int64)
        # need to add normalization

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["las_id"] = las_id
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["f_scale"] = 1.0
        target["hflip"] = target["vflip"] = False

        data_dict["target"] = target
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)])

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(segment=data_dict.pop("segment"), name=data_dict.pop("name"))
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        #data_dict_list = []
        #for aug in self.aug_transform:
        #    data_dict_list.append(aug(deepcopy(data_dict)))
        data_dict_list = [data_dict]

        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])
        result_dict["fragment_list"] = fragment_list
        return result_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

