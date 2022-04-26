import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from coinstac_dinunet import COINNDataset, COINNTrainer, COINNDataHandle

from .models import ICALstm


def read_lines(file):
    return np.array([int(float(l.strip())) for l in open(file).readlines()])


class ICADataset(COINNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.labels = None
        self.data = None
        self.window_size = self.cache['window_size']
        self.window_stride = self.cache['window_stride']
        self.temporal_size = self.cache['temporal_size']
        self.num_components = self.cache['num_components']

    def _load_indices(self, files, **kw):
        data = np.load(self.path(cache_key='data_file'))
        samples_per_sub = int(self.temporal_size / self.window_size)
        self.data = np.zeros((data.shape[0], samples_per_sub, data.shape[1], self.window_size))
        for i in range(data.shape[0]):
            for j in range(samples_per_sub):
                self.data[i, j, :, :] = data[i, :, (j * self.window_stride):(j * self.window_stride) + self.window_size]

        self.indices += files

    def __getitem__(self, ix):
        data_index, y = self.indices[ix]
        return {'inputs': torch.from_numpy(self.data[data_index].copy()), 'labels': y}


class ICATrainer(COINNTrainer):
    def __init__(self, **kw):
        super().__init__(**kw)

    def _init_nn_model(self):
        self.nn['net'] = ICALstm(
            window_size=self.cache['window_size'],  # 10
            input_size=self.cache['input_size'],  # 256
            hidden_size=self.cache['hidden_size'],  # 384
            num_comps=self.cache['num_components'],  # 100
            num_cls=self.cache['num_class'],
            num_layers=self.cache.setdefault('num_layers', 1),
            bidirectional=self.cache.setdefault('bidirectional', True)
        )

    def iteration(self, batch):
        inputs, labels = batch['inputs'].to(self.device['gpu']).float(), batch['labels'].to(
            self.device['gpu']).long()
        out, h = self.nn['net'](inputs)
        prob = torch.softmax(out, 1)
        loss = F.cross_entropy(out, labels)

        _, pred = torch.max(prob, 1)
        score = self.new_metrics()
        score.add(prob[:, 1], labels)

        val = self.new_averages()
        val.add(loss.item(), len(inputs))

        return {'out': prob, 'loss': loss, 'averages': val, 'metrics': score}


class ICADataHandle(COINNDataHandle):
    def list_files(self):
        ix_and_labels = pd.read_csv(
            self.state['baseDirectory'] + os.sep + self.cache['labels_file']).values.tolist()
        return ix_and_labels
