import os

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from coinstac_dinunet import COINNDataset, COINNTrainer, COINNDataHandle
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    SpatialPadd,
    RandAffined,
    Rand3DElasticd,
    RandAdjustContrastd,
    EnsureTyped,
    ToDeviced,
)
from catalyst.contrib.utils.pandas import dataframe_to_list
from catalyst.metrics.functional._segmentation import dice

from .models import BrainVit
from .neuro.training.brain_dataset_cache import BrainCacheDataset

def read_lines(file):
    return np.array([int(float(l.strip())) for l in open(file).readlines()])


class BrainSegDataset(COINNDataset):
    def __init__(self, in_csv_train, train_subvolumes, volume_shape, subvolume_shape, **kw):
        super().__init__(**kw)
        datasets = {}
        load_image_trans = LoadImaged(keys=["images", "nii_labels"])
        add_channel = AddChanneld(keys=["images", "nii_labels"])
        pad = SpatialPadd(keys=["images", "nii_labels"], spatial_size=[256, 256, 256])
        min_max_scale = ScaleIntensityd(keys=["images"])
        transform_chance = .5
        affine = RandAffined(keys=['images', 'nii_labels'], prob=transform_chance,
                            rotate_range=0.8,
                            translate_range=.3,
                            shear_range=0.1,
                            scale_range=0.2,
                            cache_grid=True,
                            device=torch.device("cuda"))
        to_device = ToDeviced(keys=['images', 'nii_labels'], device='cuda')


        typed = EnsureTyped(keys=["images", "nii_labels"])
        train_transforms = Compose([load_image_trans, add_channel, #min_max_scale, add_channel,
                                    typed,
                                    pad])#, to_device])

        train_rand_transforms = Compose([to_device, affine])#, elastic])

        mode="train"
        source = in_csv_train
        
        n_subvolumes = train_subvolumes
        shuffle_flag = True

        if source is not None and len(source) > 0:
            data = dataframe_to_list(pd.read_csv(source))
            datasets[mode] = BrainCacheDataset(
                data,
                list_shape=volume_shape,
                list_sub_shape=subvolume_shape,
                n_subvolumes=n_subvolumes,
                mode=mode,
                num_workers=48,
                copy_cache=False,
                transform=train_transforms,
                rand_transform=train_rand_transforms,
            )
        self.data = datasets['train']

    def __getitem__(self, ix):
        image, target = self.data[ix]
        return {'inputs': image, 'labels': target}


class BrainVitTrainer(COINNTrainer):
    def __init__(self, **kw):
        super().__init__(**kw)

    def _init_nn_model(self):
        self.nn['net'] = BrainVit(
            num_classes=self.cache.setdefault('num_class', 3)
        )

    def iteration(self, batch):
        inputs, labels = batch['inputs'].to(self.device['gpu']).float(), batch['labels'].to(self.device['gpu']).long()
        y_hat = self.nn['net'](inputs)
        loss = F.cross_entropy(y_hat, labels)
        one_hot_targets = (
            torch.nn.functional.one_hot(labels, self.n_classes)
            .permute(0, 4, 1, 2, 3)
        )

        macro_dice = dice(F.softmax(y_hat), one_hot_targets, mode="macro")

        return {'out': y_hat, 'loss': loss, 'averages': loss.item()/y_hat.shape[0], 'metrics': macro_dice}


class BrainSegDataHandle(COINNDataHandle):
    def list_files(self):
        source = self.state['baseDirectory'] + os.sep + self.cache['labels_file']
        ix = dataframe_to_list(pd.read_csv(source))
        return ix
