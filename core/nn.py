import json
import random
from os import sep

import numpy as np
import torch
from torch.nn import functional as F

from core import utils as ut
from core.measurements import new_metrics, Avg
from core.utils import save_logs


def next_iter(cache):
    out = {}
    cache['cursor'] += cache['batch_size']
    if cache['cursor'] >= cache['data_len']:
        out['mode'] = 'val_waiting'
        cache['cursor'] = 0
        random.shuffle(cache['data_indices'])
    return out


def next_epoch(cache):
    """
    Transition to next epoch after validation.
    It will set 'train_waiting' status if we need more training
    Else it will set 'test' status
    """
    out = {}
    cache['epoch'] += 1
    if cache['epoch'] - cache.get('best_epoch', cache['epoch']) \
            >= cache['patience'] or cache['epoch'] >= cache['epochs']:
        out['mode'] = 'test'
    else:
        cache['cursor'] = 0
        out['mode'] = 'train_waiting'
        random.shuffle(cache['data_indices'])

    out['train_log'] = cache['train_log']
    cache['train_log'] = []
    return out


def init_dataset(cache, state, datset_cls, min_batch_size=4):
    """
    Parse and load dataset and save to cache:
    so that in next global iteration we dont have to do that again.
    The data IO depends on use case-For a instance, if your data can fit in RAM, you can load
     and save the entire dataset in cache. But, in general,
     it is better to save indices in cache, and load only a mini-batch at a time
     (logic in __nextitem__) of the data loader.
    """
    dataset = datset_cls(cache=cache, state=state, mode=cache['mode'])
    split = json.loads(open(cache['split_dir'] + sep + cache['split_file']).read())
    dataset.load_indices(files=split['train'])

    cache['data_indices'] = dataset.indices
    if len(dataset) % cache['batch_size'] >= min_batch_size:
        cache['data_len'] = len(dataset)
    else:
        cache['data_len'] = (len(dataset) // cache['batch_size']) * cache['batch_size']


def iteration(cache, batch, nn):
    inputs, labels = batch['inputs'].to(nn['device']).float(), batch['labels'].to(nn['device']).long()
    indices = batch['ix'].to(nn['device']).long()

    out = F.log_softmax(nn['model'](inputs), 1)
    wt = torch.randint(1, 101, (2,)).to(nn['device']).float()
    loss = F.nll_loss(out, labels, weight=wt)

    _, predicted = torch.max(out, 1)
    score = new_metrics(cache['num_class'])
    score.add(predicted, labels)
    val = Avg()
    val.add(loss.item(), len(inputs))
    return {'out': out, 'loss': loss, 'avg_loss': val, 'score': score, 'prediction': predicted,
            'indices': indices}


def get_next_batch(cache, state, dataset_cls, mode='train'):
    dataset = dataset_cls(cache=cache, state=state, mode=mode, indices=cache['data_indices'][cache['cursor']:])
    loader = dataset.loader(batch_size=cache['batch_size'], num_workers=cache.get('num_workers', 0),
                            pin_memory=cache.get('pin_memory', True))
    return next(loader.__iter__())


def backward(cache, state, nn, dataset_cls):
    out = {}
    batch = get_next_batch(cache, state, dataset_cls)
    it = iteration(cache, batch, nn)
    it['loss'].backward()
    out['grads_file'] = 'grads.npy'
    grads = [p.grad.detach().cpu().numpy() for p in nn['model'].parameters()]
    np.save(state['transferDirectory'] + sep + out['grads_file'], np.array(grads))
    cache['train_log'].append([vars(it['avg_loss']), vars(it['score'])])
    return out


def step(input, state, nn):
    avg_grads = np.load(state['baseDirectory'] + sep + input['avg_grads'], allow_pickle=True)
    for i, param in enumerate(nn['model'].parameters()):
        param.grad = torch.tensor(avg_grads[i]).to(nn['device'])
    nn['optimizer'].step()


def get_predictions(dataset, it):
    predictions = []
    for ix, pred, out in zip(it['indices'].cpu().numpy(), it['prediction'], it['out'][:, 1].exp()):
        file, label = dataset.indices[int(ix)]
        predictions.append([file, label, pred.item(), round(out.item(), 3)])
    return predictions


def save_predictions(cache, predictions):
    cache['predictions'] = predictions
    save_logs(cache, file_keys=['predictions'], log_dir=cache['log_dir'])


def evaluation(cache, state, split_key, nn, dataset_cls, save_preds=None):
    nn['model'].eval()
    avg = Avg()
    eval_score = new_metrics(cache['num_class'])
    eval_predictions = ['file,true_label,prediction,probability']
    with torch.no_grad(), open(cache['split_dir'] + sep + cache['split_file']) as split_file:
        eval_dataset = dataset_cls(cache=cache, state=state, mode=split_key)
        split = json.loads(split_file.read())
        eval_dataset.load_indices(files=split[split_key])
        eval_dataloader = eval_dataset.loader(shuffle=False,
                                              batch_size=ut.get_safe_batch_size(cache['batch_size'], len(eval_dataset)),
                                              num_workers=cache.get('num_workers', 0),
                                              pin_memory=cache.get('pin_memory', True))
        for batch in eval_dataloader:
            it = iteration(cache, batch, nn)
            avg.accumulate(it['avg_loss'])
            eval_score.accumulate(it['score'])
            if save_preds:
                eval_predictions += get_predictions(eval_dataset, it)

    if save_preds:
        save_predictions(cache, eval_predictions)
    return avg, eval_score


def train_n_eval(nn, cache, input, state, dataset_cls, nxt_phase):
    out = {}
    out['mode'] = input['global_modes'].get(state['clientId'], cache['mode'])
    ut.load_checkpoint(cache, nn, id=cache['current_nn_state'])
    nn['model'].train()
    nn['optimizer'].zero_grad()

    if input.get('save_current_as_best'):
        ut.save_checkpoint(cache, nn, id=cache['best_nn_state'])
        cache['best_epoch'] = cache['epoch']

    if any(m == 'train' for m in input['global_modes'].values()):
        """
        All sites must begin/resume the training the same time.
        To enforce this, we have a 'val_waiting' status. Lagged sites will go to this status, and reshuffle the data,
         take part in the training with everybody until all sites go to 'val_waiting' status.
        """
        out.update(**backward(cache, state, nn, dataset_cls=dataset_cls))
        out.update(**next_iter(cache))

    if input.get('avg_grads'):
        step(input, state, nn)
        ut.save_checkpoint(cache, nn, id=cache['current_nn_state'])

    if out['mode'] == 'validation':
        """
        Once all sites are in 'val_waiting' status, remote issues 'validation' signal. 
        Once all sites run validation phase, they go to 'train_waiting' status. 
        Once all sites are in this status, remote issues 'train' signal and all sites reshuffle the indices and resume training.
        We send the confusion matrix to the remote to accumulate global score for model selection.
        """
        avg, scores = evaluation(cache, state, 'validation', nn, dataset_cls, save_preds=False)
        out['validation_log'] = [vars(avg), vars(scores)]
        out.update(**next_epoch(cache))

    elif out['mode'] == 'test':
        """
        We send confusion matrix to remote for global test score.
        """
        ut.load_checkpoint(cache, nn, id=cache['best_nn_state'])
        avg, scores = evaluation(cache, state, 'test', nn, dataset_cls, save_preds=True)
        out['test_log'] = [vars(avg), vars(scores)]
        out['mode'] = cache['_mode_']
        nxt_phase = 'next_run_waiting'
    return out, nxt_phase
