import os

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from coinstac_dinunet import COINNDataset, COINNTrainer, COINNDataHandle
from coinstac_dinunet.metrics import Prf1a

from .models import ICALstm


# input_size=256,
# hidden_size=256,
# num_layers=1,
# num_cls=2,
# bidirectional=True,
# proj_size=64, num_comps=53,
# window_size=20,
# seq_len=7

def read_lines(file):
    return np.array([int(float(l.strip())) for l in open(file).readlines()])


class ICADataset(COINNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.labels = None
        self.data = None
        self.full_comp = 100
        self.spatial_dim = 140
        self.window_size = 20
        self.window_stride = 10

    def load_index(self, ix):
        if self.data is None:
            self.labels = pd.DataFrame(self.state['baseDirectory'] + os.sep + self.cache['labels_file'])
            self.labels = self.labels.set_index('data_index')

            hf = h5py.File(self.path(cache_key='data_file'), "r")
            data = np.array(hf.get(self.cache['h5py_key']))
            data = data.reshape((data.shape[0], self.full_comp, self.spatial_dim))

            use_ix = read_lines(self.path(cache_key='components_file'))
            data = data[:, use_ix, :]

            unfold = nn.Unfold(kernel_size=(1, self.window_size), stride=(1, self.window_stride))
            data = unfold(data.unsqueeze(2)).reshape(157, -1, 53, self.window_stride)

            assert (data.shape[1] == self.cache['seq_len']), \
                f"Sequence len did not match: {data.shape[1]} vs {self.cache['seq_len']}"
            self.data = data

        y = self.labels.loc[ix][0]

        """int64 could not be json serializable.  """
        self.indices.append([ix, int(y)])

    def __getitem__(self, ix):
        data_index, y = self.indices[ix]
        return {'inputs': self.data[data_index].clone(), 'labels': y}


class ICATrainer(COINNTrainer):
    def __init__(self, **kw):
        super().__init__(**kw)

    def _init_nn_model(self):
        self.nn['net'] = ICALstm(
            num_layers=self.cache.setdefault('num_layers', 1),
            input_size=self.cache.setdefault('input_size', 512),
            seq_len=self.cache.setdefault('seq_len', 13),
            hidden_size=self.cache.setdefault('hidden_size', 384),
            proj_size=self.cache.setdefault('proj_size', 128)
        )

    def iteration(self, batch):
        inputs, labels = batch['inputs'].to(self.device['gpu']).float(), batch['labels'].to(self.device['gpu']).long()

        out = F.log_softmax(self.nn['net'](inputs), 1)
        loss = F.nll_loss(out, labels)

        _, pred = torch.max(out, 1)
        score = self.new_metrics()
        score.add(pred, labels)

        val = self.new_averages()
        val.add(loss.item(), len(inputs))

        return {'out': out, 'loss': loss, 'averages': val, 'metrics': score, 'prediction': pred}

    def _set_monitor_metric(self):
        self.cache['monitor_metric'] = 'f1', 'maximize'

    def _set_log_headers(self):
        self.cache['log_header'] = 'loss|accuracy,f1'

    def new_metrics(self):
        return Prf1a()


class ICADataHandle(COINNDataHandle):
    def list_files(self):
        ix = list(
            pd.DataFrame(self.state['baseDirectory'] + os.sep + self.cache['labels_file'])['data_index'].tolist()
        )
        return ix
