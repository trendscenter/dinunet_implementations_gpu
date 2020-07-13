#!/usr/bin/python
import json
import os
import random
import sys
from os import sep

from classification import train, evaluation
from core import utils as ut
from core.datautils import init_k_folds


# import pydevd_pycharm
#
# pydevd_pycharm.settrace('172.17.0.1', port=8881, stdoutToServer=True, stderrToServer=True, suspend=False)


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
        """
        Generate folds as specified. 
        """
        cache.update(**input)
        out.update(**init_k_folds(cache, state))
        cache['_mode_'] = input['mode']
        out['mode'] = cache['mode']
        splits = os.listdir(state['baseDirectory'] + sep + cache['split_dir'])

    if nxt_phase == 'init_nn':
        """
        Initialize neural network/optimizer and GPUs
        """
        cache.update(**input['run'][state['clientId']], epoch=0, cursor=0, train_log=[])
        cache['split_file'] = cache['splits'][cache['split_ix']]
        cache['log_dir'] = state['outputDirectory'] + sep + cache['id'] + sep + cache['split_file'].split('.')[0]
        os.makedirs(cache['log_dir'], exist_ok=True)
        cache['current_nn_state'] = 'current.nn.pt'
        cache['best_nn_state'] = 'best.nn.pt'
        nn = ut.init_nn(cache, init_weights=True)
        ut.save_checkpoint(cache, nn['model'], nn['optimizer'], id=cache['current_nn_state'])
        ut.init_dataset(cache, state)
        nxt_phase = 'computation'

    if nxt_phase == 'computation':
        """
        Train/validation and test phases
        """
        nn = ut.init_nn(cache, init_weights=False)
        out['mode'] = input['global_modes'].get(state['clientId'], cache['mode'])

        if input.get('save_current_as_best'):
            ut.load_checkpoint(cache, nn['model'], nn['optimizer'], id=cache['current_nn_state'])
            ut.save_checkpoint(cache, nn['model'], nn['optimizer'], id=cache['best_nn_state'])

        if out['mode'] in ['train', 'val_waiting']:
            """
            All sites must begin/resume the training the same time.
            To enforce this, we have a 'val_waiting' status. Lagged sites will go to this status, and reshuffle the data,
             take part in the training with everybody until all sites go to 'val_waiting' status.
            """
            ut.load_checkpoint(cache, nn['model'], nn['optimizer'], id=cache['current_nn_state'])
            out.update(**train(cache, input, state, nn['model'], nn['optimizer'], device=nn['device']))
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
            avg, scores = evaluation(cache, state, nn['model'], split_key='validation', device=nn['device'])
            out['validation_log'] = [vars(avg), vars(scores)]
            out.update(**next_epoch(cache))

        elif out['mode'] == 'test':
            """
            We send confusion matrix to remote for global test score.
            """
            ut.load_checkpoint(cache, nn['model'], nn['optimizer'], id=cache['best_nn_state'])
            avg, scores = evaluation(cache, state, nn['model'], split_key='validation', device=nn['device'])
            out['test_log'] = [vars(avg), vars(scores)]
            out['mode'] = cache['_mode_']
            nxt_phase = 'next_run_waiting'

    out['phase'] = nxt_phase
    output = json.dumps({'output': out, 'cache': cache})
    sys.stdout.write(output)
    args, cache, input, state, out, output = [None] * 6
