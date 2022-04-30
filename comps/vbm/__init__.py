import os

import nibabel as ni
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchio as tio
from coinstac_dinunet import COINNDataset, COINNTrainer, COINNDataHandle

from .models import VBMNet


class VBMDataset(COINNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.labels = None
        self.transform = tio.Compose([tio.RandomFlip(axes=(0, 1, 2))])

    def load_index(self, file):
        if self.labels is None:
            self.labels = pd.read_csv(self.state['baseDirectory'] + os.sep + self.cache["labels_file"])
            if self.cache.get('data_column') in self.labels.columns:
                self.labels = self.labels.set_index(self.cache['data_column'])

        y = self.labels.loc[file][self.cache['labels_column']]

        if isinstance(y, str):
            y = int(y.strip().lower() == 'true')

        """
        int64 could not be json serializable.
        """
        self.indices.append([file, int(y)])

    def __getitem__(self, ix):
        file, y = self.indices[ix]
        data_dir = self.path()
        nif = np.array(ni.load(data_dir + os.sep + file).dataobj)
        nif[nif < 0.05] = 0

        # mean = np.load(self.state[site]['baseDirectory'] + os.sep + f"{site}_mean.npy")
        # std = np.load(self.state[site]['baseDirectory'] + os.sep + f"{site}_std.npy")
        # std[nif == 0] = 1
        # nif = (nif - mean) / std

        nif = nif[None, :]
        if self.mode == 'train':
            nif = self.transform(nif)
        return {'inputs': nif.copy(), 'labels': y, 'ix': ix}


class VBMTrainer(COINNTrainer):
    def __init__(self, **kw):
        super().__init__(**kw)

    def _init_nn_model(self):
        self.nn['net'] = VBMNet(num_channels=self.cache['input_channel'],
                                num_classes=self.cache['num_class'], b_mul=self.cache.get('model_scale', 1))

    def iteration(self, batch):
        inputs, labels = batch['inputs'].to(self.device['gpu']).float(), batch['labels'].to(self.device['gpu']).long()
        indices = batch['ix'].to(self.device['gpu']).long()

        out = F.log_softmax(self.nn['net'](inputs), 1)
        loss = F.nll_loss(out, labels)

        _, predicted = torch.max(out, 1)
        score = self.new_metrics()
        score.add(predicted, labels)
        val = self.new_averages()
        val.add(loss.item(), len(inputs))
        return {'out': out, 'loss': loss, 'averages': val, 'metrics': score, 'prediction': predicted,
                'indices': indices}


class VBMDataHandle(COINNDataHandle):
    def list_files(self):
        df = pd.read_csv(self.state['baseDirectory'] + os.sep + self.cache["labels_file"])
        if self.cache.get('data_column') in df.columns:
            df = df.set_index(self.cache['data_column'])
        return list(df.index)
