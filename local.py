#!/usr/bin/python
import json
import os
import random
import sys
from os import sep

import torch

from classification import FreeSurferDataset, train, evaluation
from core import utils
from core.models import MSANNet


# pydevd_pycharm.settrace('172.17.0.1', port=8881, stdoutToServer=True, stderrToServer=True, suspend=False)


def init_nn(cache, init_weights=False):
    device = cache.get('device', 'cpu')
    model = MSANNet(in_size=cache['input_size'], hidden_sizes=cache['hidden_sizes'],
                    out_size=cache['num_class'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cache['learning_rate'])
    if init_weights:
        torch.manual_seed(cache['seed'])
        utils.initialize_weights(model)
    return {'device': device, 'model': model.to(device), 'optimizer': optimizer}


def init_dataset(cache, state):
    dataset = FreeSurferDataset(files_dir=state['baseDirectory'] + sep + cache['data_dir'],
                                labels_file=state['baseDirectory'] + sep + cache['label_dir'],
                                mode=cache['mode'])
    split = json.loads(
        open(state['baseDirectory'] + sep + cache['split_dir'] + sep + cache['split_file']).read())
    dataset.load_indices(files=split['train'])
    cache['data_indices'] = dataset.indices
    if len(dataset) % cache['batch_size'] >= 4:
        cache['data_len'] = len(dataset)
    else:
        cache['data_len'] = (len(dataset) // cache['batch_size']) * cache['batch_size']


def next_iter(cache):
    out = {}
    cache['cursor'] += cache['batch_size']
    if cache['cursor'] >= cache['data_len']:
        out['mode'] = 'val_waiting'
        cache['cursor'] = 0
        random.shuffle(cache['data_indices'])

    return out


def next_epoch(cache):
    out = {}
    cache['epoch'] += 1
    if cache['epoch'] >= cache['epochs']:
        out['mode'] = 'test'
    else:
        cache['cursor'] = 0
        out['mode'] = 'train_waiting'
        random.shuffle(cache['data_indices'])

    out['train_log'] = cache['train_log']
    cache['train_log'] = []
    return out


if __name__ == "__main__":
    args = json.loads(sys.stdin.read())
    cache = args['cache']
    input = args['input']
    state = args['state']
    out = {}

    nxt_phase = input.get('phase', 'init_runs')
    if nxt_phase == 'init_runs':
        cache.update(**input)
        cache['_mode_'] = input['mode']
        out['mode'] = cache['mode']
        splits = os.listdir(state['baseDirectory'] + sep + cache['split_dir'])
        cache['splits'] = dict(zip(range(len(splits)), splits))
        out['splits'] = {}
        for i, sp in cache['splits'].items():
            sp = json.loads(open(f"{state['baseDirectory'] + sep + cache['split_dir']}/{sp}").read())
            out['splits'][i] = len(sp['train'])

    if nxt_phase == 'init_nn':
        cache.update(**input['nn'], **input['run'][state['clientId']], epoch=0, cursor=0, train_log=[])
        cache['split_file'] = cache['splits'][cache['split_ix']]
        cache['log_dir'] = state['outputDirectory'] + sep + cache['eid'] + sep + cache['split_file'].split('.')[0]
        os.makedirs(cache['log_dir'], exist_ok=True)
        cache['current_nn_state'] = 'current.nn.pt'
        cache['best_nn_state'] = 'best.nn.pt'
        nn = init_nn(cache, init_weights=True)
        utils.save_checkpoint(cache, nn['model'], nn['optimizer'], id=cache['current_nn_state'])
        init_dataset(cache, state)
        nxt_phase = 'computation'

    if nxt_phase == 'computation':
        nn = init_nn(cache, init_weights=False)
        out['mode'] = input['global_modes'].get(state['clientId'], cache['mode'])

        if input.get('save_current_as_best'):
            utils.load_checkpoint(cache, nn['model'], nn['optimizer'], id=cache['current_nn_state'])
            utils.save_checkpoint(cache, nn['model'], nn['optimizer'], id=cache['best_nn_state'])

        if out['mode'] in ['train', 'val_waiting']:
            utils.load_checkpoint(cache, nn['model'], nn['optimizer'], id=cache['current_nn_state'])
            out.update(**train(cache, input, state, nn['model'], nn['optimizer'], device=nn['device']))
            utils.save_checkpoint(cache, nn['model'], nn['optimizer'], id=cache['current_nn_state'])
            out.update(**next_iter(cache))

        elif out['mode'] == 'validation':
            utils.load_checkpoint(cache, nn['model'], nn['optimizer'], id=cache['current_nn_state'])
            avg, scores = evaluation(cache, state, nn['model'], split_key='validation', device=nn['device'])
            out['validation_log'] = [vars(avg), vars(scores)]
            out.update(**next_epoch(cache))

        elif out['mode'] == 'test':
            utils.load_checkpoint(cache, nn['model'], nn['optimizer'], id=cache['best_nn_state'])
            avg, scores = evaluation(cache, state, nn['model'], split_key='validation', device=nn['device'])
            out['test_log'] = [vars(avg), vars(scores)]
            out['mode'] = cache['_mode_']
            nxt_phase = 'next_run_waiting'

    out['phase'] = nxt_phase
    output = json.dumps({'output': out, 'cache': cache})
    sys.stdout.write(output)
    args, cache, input, state, out, output = [None] * 6
