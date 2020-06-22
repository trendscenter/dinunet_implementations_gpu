#!/usr/bin/env python3
import json
import os
from os import sep

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn


def save_checkpoint(cache, model, optimizer, id):
    chk = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    torch.save(chk, cache['log_dir'] + sep + id)


def load_checkpoint(cache, model, optimizer, id):
    checkpoint = torch.load(cache['log_dir'] + sep + id)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def save_logs(cache, plot_keys=[], file_keys=[], num_points=51, log_dir=None):
    plt.switch_backend('agg')
    plt.rcParams["figure.figsize"] = [16, 9]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    for k in plot_keys:
        data = cache.get(k, [])

        if len(data) == 0:
            continue

        df = pd.DataFrame(data[1:], columns=data[0].split(','))

        if len(df) == 0:
            continue

        for col in df.columns:
            if max(df[col]) > 1:
                df[col] = scaler.fit_transform(df[[col]])

        rollin_window = max(df.shape[0] // num_points + 1, 3)
        rolling = df.rolling(rollin_window, min_periods=1).mean()
        ax = df.plot(x_compat=True, alpha=0.1, legend=0)
        rolling.plot(ax=ax, title=k.upper())

        plt.savefig(log_dir + os.sep + k + '.png')
        plt.close('all')

    for fk in file_keys:
        with open(log_dir + os.sep + f'{fk}.csv', 'w') as file:
            for line in cache[fk] if any(isinstance(ln, list)
                                         for ln in cache[fk]) else [cache[fk]]:
                if isinstance(line, list):
                    file.write(','.join([str(s) for s in line]) + '\n')
                else:
                    file.write(f'{line}\n')


def fmt(*args):
    return ','.join(str(s) for s in args)


def create_k_fold_splits(files, k=0, save_to_dir=None, shuffle_files=True):
    from random import shuffle
    from itertools import chain
    import numpy as np

    if shuffle_files:
        shuffle(files)

    ix_splits = np.array_split(np.arange(len(files)), k)
    for i in range(len(ix_splits)):
        test_ix = ix_splits[i].tolist()
        val_ix = ix_splits[(i + 1) % len(ix_splits)].tolist()
        train_ix = [ix for ix in np.arange(len(files)) if ix not in test_ix + val_ix]

        splits = {'train': [files[ix] for ix in train_ix],
                  'validation': [files[ix] for ix in val_ix],
                  'test': [files[ix] for ix in test_ix]}

        print('Valid:', set(files) - set(list(chain(*splits.values()))) == set([]))
        if save_to_dir:
            f = open(save_to_dir + os.sep + 'SPLIT_' + str(i) + '.json', "w")
            f.write(json.dumps(splits))
            f.close()
        else:
            return splits


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
