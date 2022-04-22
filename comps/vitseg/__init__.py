import os
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from coinstac_dinunet import COINNDataset, COINNTrainer, COINNDataHandle
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    SpatialPadd,
    RandAffined,
    EnsureTyped,
    ToDeviced,
)

from .models import BrainVit
from .neuro.training.brain_dataset_cache import BrainCacheDataset


def read_lines(file):
    return np.array([int(float(l.strip())) for l in open(file).readlines()])


class BrainSegDataset(COINNDataset):
    def __init__(self):
        super().__init__()
        self.train_subvolumes = self.cache['train_subvolumes']
        self.volume_shape = self.cache['volume_shape']
        self.subvolume_shape = self.cache['subvolume_shape']
        self._transforms = self.transforms()
        self.monai_dataset = None

    def transforms(self, **kw):
        load_image_trans = LoadImaged(keys=["images", "nii_labels"])
        add_channel = AddChanneld(keys=["images", "nii_labels"])
        pad = SpatialPadd(keys=["images", "nii_labels"], spatial_size=[256, 256, 256])
        affine = RandAffined(keys=['images', 'nii_labels'],
                             prob=0.5,
                             rotate_range=0.8,
                             translate_range=.3,
                             shear_range=0.1,
                             scale_range=0.2,
                             cache_grid=True,
                             device=torch.device("cuda"))
        to_device = ToDeviced(keys=['images', 'nii_labels'], device='cuda')

        typed = EnsureTyped(keys=["images", "nii_labels"])
        train_transforms = Compose([load_image_trans, add_channel,  # min_max_scale, add_channel,
                                    typed,
                                    pad])  # , to_device])

        train_rand_transforms = Compose([to_device, affine])  # , elastic])

        return {'train': train_transforms, 'rand_train': train_rand_transforms}

    def _load_indices(self, files, **kw):
        # Files will be a list as : [[file_path,label_path],[],[],....]
        self.monai_dataset = BrainCacheDataset(
            files,
            list_shape=self.volume_shape,
            list_sub_shape=self.subvolume_shape,
            n_subvolumes=self.train_subvolumes,
            mode=self.mode,
            num_workers=self.cache['num_workers'],
            copy_cache=False,
            transform=self._transforms['train'],
            rand_transform=self._transforms['rand_train'],
        )

    def __getitem__(self, ix):
        """returns a tuple"""
        return self.monai_dataset[ix]


class BrainVitTrainer(COINNTrainer):
    def __init__(self, **kw):
        super().__init__(**kw)

    def _init_nn_model(self):
        self.nn['net'] = BrainVit(n_classes=self.cache['num_class'])

    def iteration(self, batch):
        inputs, labels = batch[0].to(self.device['gpu']).float(), batch[1].to(self.device['gpu']).long()
        y_hat = self.nn['net'](inputs)
        loss = F.cross_entropy(y_hat, labels)
        y_prob = F.softmax(y_hat, 1)

        avg = self.new_averages()
        avg.add(loss.item(), len(inputs))

        metrics = self.new_metrics()
        metrics.add(y_prob, labels)

        return {'out': y_hat, 'loss': loss, 'averages': avg, "metrics": metrics}


class BrainSegDataHandle(COINNDataHandle):
    def prepare_data(self):
        with zipfile.ZipFile(self.state['baseDirectory'] + os.sep + self.cache['data_zip_file'], 'r') as zip_ref:
            zip_ref.extractall(self.state['outputDirectory'] + os.sep + 'data')
        return super(BrainSegDataHandle, self).prepare_data()

    def list_files(self):
        source = self.state['baseDirectory'] + os.sep + self.cache['labels_file']
        return pd.read_csv(source).values.tolist()
