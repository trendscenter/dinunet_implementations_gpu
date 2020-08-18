#!/usr/bin/python
import json
import os
import shutil
import sys
from os import sep

import nibabel as ni
import pandas as pd
import torch
import itertools
import numpy as np
import random

from core import utils as ut
from core.nn import train_n_eval, init_dataset
from core.utils import init_k_folds, NNDataset, initialize_weights
from models import VBMNet


# import pydevd_pycharm
#
# pydevd_pycharm.settrace('172.17.0.1', port=8881, stdoutToServer=True, stderrToServer=True, suspend=False)


class NiftiDataset(NNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.axises = []
        axises = [0, 1]
        for i in range(len(axises)):
            self.axises += list(itertools.combinations(axises, i + 1))

    def load_indices(self, files, **kw):
        labels_file = os.listdir(self.label_dir)[0]
        labels = pd.read_csv(self.label_dir + os.sep + labels_file).set_index('niftifile')
        for file in files:
            y = labels.loc[file]['isControl']
            """
            int64 could not be json serializable.
            """
            self.indices.append([file, int(y)])

    def __getitem__(self, ix):
        file, y = self.indices[ix]
        nif = np.array(ni.load(self.data_dir + sep + file).dataobj)
        nif[nif < 0.05] = 0
        # if random.uniform(0, 1) > 0.5:
        #     nif = np.flip(nif, random.choice(self.axises))
        return {'inputs': torch.tensor(nif.copy()[None, :]), 'labels': torch.tensor(y), 'ix': torch.tensor(ix)}


def init_nn(cache, state, init_weights=False):
    """
    Initialize neural network/optimizer with locked parameters(check on remote script for that).
    Also detect and assign specified GPUs.
    @note Works only with one GPU per site at the moment.
    """
    device = ut.get_cuda_device(cache, state)
    model = VBMNet(in_channels=cache['input_ch'], out_channels=cache['num_class'], init_features=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=cache['learning_rate'])
    if init_weights:
        torch.manual_seed(cache['seed'])
        initialize_weights(model)
    return {'device': device, 'model': model.to(device), 'optimizer': optimizer}


if __name__ == "__main__":

    args = json.loads(sys.stdin.read())
    cache = args['cache']
    input = args['input']
    state = args['state']
    out = {}

    nxt_phase = input.get('phase', 'init_runs')
    if nxt_phase == 'init_runs':
        """
        Generate folds as specified. 
        """
        cache.update(**input)
        out.update(**init_k_folds(cache, state))
        cache['_mode_'] = input['mode']
        out['mode'] = cache['mode']

    if nxt_phase == 'init_nn':
        """
        Initialize neural network/optimizer and GPUs
        """
        cache.update(**input['run'][state['clientId']], epoch=0, cursor=0, train_log=[])
        cache['split_file'] = cache['splits'][cache['split_ix']]
        cache['log_dir'] = state['outputDirectory'] + sep + cache['id'] + sep + str(cache['split_ix'])
        os.makedirs(cache['log_dir'], exist_ok=True)
        cache['current_nn_state'] = 'current.nn.pt'
        cache['best_nn_state'] = 'best.nn.pt'
        nn = init_nn(cache, state, init_weights=True)
        ut.save_checkpoint(cache, nn, id=cache['current_nn_state'])
        init_dataset(cache, state, NiftiDataset)
        nxt_phase = 'computation'

    if nxt_phase == 'computation':
        """
        Train/validation and test phases
        """
        nn = init_nn(cache, state, init_weights=False)
        out_, nxt_phase = train_n_eval(nn, cache, input, state, NiftiDataset, nxt_phase)
        out.update(**out_)

    elif nxt_phase == 'success':
        """
        This phase receives global scores from the aggregator.
        """
        shutil.copy(f"{state['baseDirectory']}{sep}{input['results_zip']}.zip",
                    f"{state['outputDirectory'] + sep + cache['id']}{sep}{input['results_zip']}.zip")

    out['phase'] = nxt_phase
    output = json.dumps({'output': out, 'cache': cache})
    sys.stdout.write(output)
    args, cache, input, state, out, output = [None] * 6
