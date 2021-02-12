#!/usr/bin/python
import sys
import os
import torch
import numpy as np
import pandas as pd
import nibabel as ni
from coinstac_dinunet import COINNDataset, COINNLocal, COINNTrainer
import torch.nn.functional as F
import json
from models import VBMNet


# import pydevd_pycharm
#
# pydevd_pycharm.settrace('172.17.0.1', port=8881, stdoutToServer=True, stderrToServer=True, suspend=False)

class NiftiDataset(COINNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.labels = {}

    def load_index(self, site, file):
        if self.labels.get(site) is None:
            label_dir = self.path(site, 'label_dir')
            labels_file = os.listdir(label_dir)[0]
            self.labels[site] = pd.read_csv(label_dir + os.sep + labels_file).set_index('niftifile')
        y = self.labels[site].loc[file]['isControl']
        self.indices.append([site, file, int(y)])

    def __getitem__(self, ix):
        site, file, y = self.indices[ix]
        data_dir = self.path(site, 'data_dir')
        nif = np.array(ni.load(data_dir + os.sep + file).dataobj)
        nif[nif < 0.05] = 0
        return {'inputs': torch.tensor(nif.copy()[None, :]), 'labels': torch.tensor(y), 'ix': torch.tensor(ix)}


class NiftiTrainer(COINNTrainer):
    def __init__(self, **kw):
        super().__init__(**kw)

    def _init_nn_model(self):
        self.nn['net'] = VBMNet(num_channels=self.cache['input_ch'],
                                num_classes=self.cache['num_class'],  reduce_by=64)

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
        self.cache['log_header'] = 'loss,accuracy,f1'


if __name__ == "__main__":
    args = json.loads(sys.stdin.read())
    local = COINNLocal(cache=args['cache'], input=args['input'],
                       state=args['state'], epochs=111, patience=21, gpus=[],
                       pretrain_epochs=21, computation_id='fsv_volumes_pretrained')
    local.compute(NiftiDataset, NiftiTrainer)
    local.send()
