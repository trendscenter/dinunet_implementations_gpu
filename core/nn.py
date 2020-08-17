import json
import random
from os import sep

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


def iteration(cache, batch, model, optimizer=None, **kw):
    inputs, labels = batch['inputs'].to(kw['device']).float(), batch['labels'].to(kw['device']).long()
    indices = batch['ix'].to(kw['device']).long()

    if optimizer and model.training:
        optimizer.zero_grad()

    out = F.log_softmax(model(inputs), 1)
    loss = F.nll_loss(out, labels)

    if optimizer and model.training:
        loss.backward()
        if kw.get('avg_grad') is not None:
            for i, param in enumerate(model.parameters()):
                tensor = kw.get('avg_grad')[i].to(kw['device'])
                param.grad.data = torch.autograd.Variable(tensor)
            optimizer.step()

    _, predicted = torch.max(out, 1)
    score = new_metrics(cache['num_class'])
    score.add(predicted, labels)
    val = Avg()
    val.add(loss.item(), len(inputs))

    return {'out': out, 'loss': val, 'score': score, 'prediction': predicted,
            'indices': indices}


def get_next_batch(cache, state, dataset_cls, mode='train'):
    dataset = dataset_cls(cache=cache, state=state, mode=mode, indices=cache['data_indices'][cache['cursor']:])
    loader = dataset.loader(batch_size=cache['batch_size'], num_workers=cache.get('num_workers', 0),
                            pin_memory=cache.get('pin_memory', True))
    return next(loader.__iter__())


def train(cache, input, state, model, optimizer, dataset_cls, **kw):
    out = {}
    model.train()
    model = model.to(kw['device'])
    avg_grads = torch.load(state['baseDirectory'] + sep + input['avg_grads_file']) \
        if input.get('avg_grads_file') else None
    batch = get_next_batch(cache, state, dataset_cls)
    it = iteration(cache, batch, model, optimizer, avg_grad=avg_grads, device=kw['device'])
    cache['train_log'].append([vars(it['loss']), vars(it['score'])])
    out['grads_file'] = 'grads.tar'
    grads = [p.grad for p in model.parameters()]
    torch.save(grads, state['transferDirectory'] + sep + out['grads_file'])
    return out


def get_predictions(dataset, it):
    predictions = []
    for ix, pred, out in zip(it['indices'].cpu().numpy(), it['prediction'], it['out'][:, 1].exp()):
        file, label = dataset.indices[int(ix)]
        predictions.append([file, label, pred.item(), round(out.item(), 3)])
    return predictions


def save_predictions(cache, predictions):
    cache['predictions'] = predictions
    save_logs(cache, file_keys=['predictions'], log_dir=cache['log_dir'])


def evaluation(cache, state, model, split_key, dataset_cls, **kw):
    model.eval()
    model = model.to(kw['device'])

    avg = Avg()
    eval_score = new_metrics(cache['num_class'])
    eval_predictions = ['file,true_label,prediction,probability']
    with torch.no_grad(), open(cache['split_dir'] + sep + cache['split_file']) as split_file:
        eval_dataset = dataset_cls(cache=cache, state=state, mode=split_key)
        split = json.loads(split_file.read())
        eval_dataset.load_indices(files=split[split_key])
        eval_dataloader = eval_dataset.loader(shuffle=False, batch_size=cache['batch_size'],
                                              num_workers=cache.get('num_workers', 0),
                                              pin_memory=cache.get('pin_memory', True))
        for batch in eval_dataloader:
            it = iteration(cache, batch, model, device=kw['device'])
            avg.accumulate(it['loss'])
            eval_score.accumulate(it['score'])
            if kw.get('save_predictions'):
                eval_predictions += get_predictions(eval_dataset, it)

    if kw.get('save_predictions'):
        save_predictions(cache, eval_predictions)
    return avg, eval_score


def train_n_eval(nn, cache, input, state, dataset_cls, nxt_phase):
    out = {}
    out['mode'] = input['global_modes'].get(state['clientId'], cache['mode'])

    if input.get('save_current_as_best'):
        ut.load_checkpoint(cache, nn['model'], nn['optimizer'], id=cache['current_nn_state'])
        ut.save_checkpoint(cache, nn['model'], nn['optimizer'], id=cache['best_nn_state'])
        cache['best_epoch'] = cache['epoch']

    if out['mode'] in ['train', 'val_waiting']:
        """
        All sites must begin/resume the training the same time.
        To enforce this, we have a 'val_waiting' status. Lagged sites will go to this status, and reshuffle the data,
         take part in the training with everybody until all sites go to 'val_waiting' status.
        """
        ut.load_checkpoint(cache, nn['model'], nn['optimizer'], id=cache['current_nn_state'])
        out.update(
            **train(cache, input, state, nn['model'], nn['optimizer'],
                    device=nn['device'], dataset_cls=dataset_cls))
        ut.save_checkpoint(cache, nn['model'], nn['optimizer'], id=cache['current_nn_state'])
        out.update(**next_iter(cache))

    elif out['mode'] == 'validation':
        """
        Once all sites are in 'val_waiting' status, remote issues 'validation' signal. 
        Once all sites run validation phase, they go to 'train_waiting' status. 
        Once all sites are in this status, remote issues 'train' signal and all sites reshuffle the indices and resume training.
        We send the confusion matrix to the remote to accumulate global score for model selection.
        """
        ut.load_checkpoint(cache, nn['model'], nn['optimizer'], id=cache['current_nn_state'])
        avg, scores = evaluation(cache, state, nn['model'],
                                 split_key='validation', device=nn['device'], dataset_cls=dataset_cls)
        out['validation_log'] = [vars(avg), vars(scores)]
        out.update(**next_epoch(cache))

    elif out['mode'] == 'test':
        """
        We send confusion matrix to remote for global test score.
        """
        ut.load_checkpoint(cache, nn['model'], nn['optimizer'], id=cache['best_nn_state'])
        avg, scores = evaluation(cache, state, nn['model'], split_key='validation', device=nn['device'],
                                 save_predictions=True, dataset_cls=dataset_cls)
        out['test_log'] = [vars(avg), vars(scores)]
        out['mode'] = cache['_mode_']
        nxt_phase = 'next_run_waiting'
    return out, nxt_phase
