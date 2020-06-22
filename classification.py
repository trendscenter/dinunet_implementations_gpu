# !/usr/bin/python

import json
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from core import data_parser
from core.torchutils import NNDataLoader

sep = os.sep
from core.measurements import new_metrics, Avg
import numpy as np


class FreeSurferDataset(Dataset):
    def __init__(self, **kw):
        self.files_dir = kw['files_dir']
        self.labels_dir = kw['labels_file']
        self.mode = kw['mode']
        self.indices = []

    def load_indices(self, files, **kw):
        labels_file = os.listdir(self.labels_dir)[0]
        labels = pd.read_csv(self.labels_dir + os.sep + labels_file).set_index('freesurferfile')
        for file in files:
            y = labels.loc[file]['label']
            """
            int64 could not be json serializable.
            """
            self.indices.append([file, int(y)])

    def __getitem__(self, ix):
        file, y = self.indices[ix]
        data, errors = data_parser.parse_subj_volume_files(self.files_dir, [file])
        x = data.iloc[0].values
        return {'inputs': torch.tensor(x), 'labels': torch.tensor(y)}

    def __len__(self):
        return len(self.indices)

    def get_loader(self, shuffle=False, batch_size=None, num_workers=0, pin_memory=True, **kw):
        return NNDataLoader.get_loader(dataset=self, shuffle=shuffle, batch_size=batch_size,
                                       num_workers=num_workers, pin_memory=pin_memory, **kw)


def get_next_batch(cache, state):
    dataset = FreeSurferDataset(files_dir=state['baseDirectory'] + sep + cache['data_dir'],
                                labels_file=state['baseDirectory'] + sep + cache['label_dir'], mode=cache['mode'])
    dataset.indices = cache['data_indices'][cache['cursor']:]
    loader = dataset.get_loader(batch_size=cache['batch_size'], num_workers=cache.get('num_workers', 0),
                                pin_memory=cache.get('pin_memory', True))
    return next(loader.__iter__())


def iteration(cache, batch, model, optimizer=None, **kw):
    inputs, labels = batch['inputs'].to(kw['device']).float(), batch['labels'].to(kw['device']).long()

    if optimizer and model.training:
        optimizer.zero_grad()

    out = F.log_softmax(model(inputs), 1)
    loss = F.nll_loss(out, labels)

    if optimizer and model.training:
        loss.backward()
        if kw.get('avg_grad') is not None:
            for i, param in enumerate(model.parameters()):
                tensor = torch.tensor(kw.get('avg_grad')[i]).to(kw['device']).float()
                param.grad.data = torch.autograd.Variable(tensor)
            optimizer.step()

    _, predicted = torch.max(out, 1)
    score = new_metrics(cache['num_class'])
    score.add(predicted, labels)
    val = Avg()
    val.add(loss.item(), len(inputs))

    return {'out': out, 'loss': val, 'score': score, 'prediction': predicted}


def train(cache, input, state, model, optimizer, **kw):
    out = {}
    model.train()
    model = model.to(kw['device'])
    avg_grads = np.load(state['baseDirectory'] + sep + input['avg_grads_file'], allow_pickle=True) \
        if input.get('avg_grads_file') else None
    batch = get_next_batch(cache, state)
    it = iteration(cache, batch, model, optimizer, avg_grad=avg_grads, device=kw['device'])
    cache['train_log'].append([vars(it['loss']), vars(it['score'])])
    out['grads_file'] = 'grads.npy'
    np.save(state['transferDirectory'] + sep + out['grads_file'],
            np.array([p.grad.numpy() for p in model.parameters()]))
    return out


def evaluation(cache, state, model, split_key, **kw):
    model.eval()
    model = model.to(kw['device'])

    avg = Avg()
    eval_score = new_metrics(cache['num_class'])
    with torch.no_grad(), open(
            state['baseDirectory'] + sep + cache['split_dir'] + sep + cache['split_file']) as split_file:
        eval_dataset = FreeSurferDataset(files_dir=state['baseDirectory'] + sep + cache['data_dir'],
                                         labels_file=state['baseDirectory'] + sep + cache['label_dir'],
                                         mode=cache['mode'])
        split = json.loads(split_file.read())
        eval_dataset.load_indices(files=split[split_key])
        eval_dataloader = eval_dataset.get_loader(shuffle=False, batch_size=cache['batch_size'],
                                                  num_workers=cache.get('num_workers', 0),
                                                  pin_memory=cache.get('pin_memory', True))
        for batch in eval_dataloader:
            it = iteration(cache, batch, model, device=kw['device'])
            avg.accumulate(it['loss'])
            eval_score.accumulate(it['score'])
    return avg, eval_score
