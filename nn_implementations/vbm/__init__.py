import os

import nibabel as ni
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchio as tio
from coinstac_dinunet import COINNDataset, COINNTrainer, COINNDataHandle
from coinstac_dinunet.metrics import Prf1a
from coinstac_dinunet.data.datautils import init_k_folds

from .models import VBMNet


class VBMDataset(COINNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.labels = {}
        self.transform = tio.Compose([tio.RandomFlip(axes=(0, 1, 2))])

    def load_index(self, site, file):
        if self.labels.get(site) is None:
            self.labels[site] = pd.DataFrame(self.inputspecs[site]['covariates']).T
        y = self.labels[site].loc[file][self.inputspecs[site]['labels_column']]

        if isinstance(y, str):
            y = int(y.strip().lower() == 'true')

        """
        int64 could not be json serializable.
        """
        self.indices.append([site, file, int(y)])

    def __getitem__(self, ix):
        site, file, y = self.indices[ix]
        data_dir = self.path(site)
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
                                num_classes=self.cache['num_class'], b_mul=self.cache['model_scale'])

    def iteration(self, batch):
        inputs, labels = batch['inputs'].to(self.device['gpu']).float(), batch['labels'].to(self.device['gpu']).long()
        indices = batch['ix'].to(self.device['gpu']).long()

        out = F.log_softmax(self.nn['net'](inputs), 1)
        wt = torch.randint(1, 101, (2,)).to(self.device['gpu']).float()
        loss = F.nll_loss(out, labels, weight=wt)

        _, predicted = torch.max(out, 1)
        score = self.new_metrics()
        score.add(predicted, labels)
        val = self.new_averages()
        val.add(loss.item(), len(inputs))
        return {'out': out, 'loss': loss, 'averages': val, 'metrics': score, 'prediction': predicted,
                'indices': indices}

    def _set_monitor_metric(self):
        self.cache['monitor_metric'] = 'f1', 'maximize'

    def _set_log_headers(self):
        self.cache['log_header'] = 'loss|accuracy,f1'

    def new_metrics(self):
        return Prf1a()


class VBMDataHandle(COINNDataHandle):
    def prepare_data(self):
        files = list(pd.DataFrame(self.cache['covariates']).T.index)
        return init_k_folds(files, self.cache, self.state)
