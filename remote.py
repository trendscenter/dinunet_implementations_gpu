#!/usr/bin/python
import datetime
import json
import os
import shutil
import sys
from itertools import repeat

import numpy as np

import core.utils
from core.measurements import Prf1a, Avg
import torch

# import pydevd_pycharm
#
# pydevd_pycharm.settrace('172.17.0.1', port=8881, stdoutToServer=True, stderrToServer=True, suspend=False)


def aggregate_sites_info(input):
    """
    Average each sites gradients and pass it to all sites.
    """
    out = {}
    grads = []
    for site, site_vars in input.items():
        grad_file = state['baseDirectory'] + os.sep + site + os.sep + site_vars['grads_file']
        grads.append(torch.load(grad_file))
    out['avg_grads_file'] = 'avg_grads.tar'
    avg_grads = []
    for layer_grad in zip(*grads):
        """
        RuntimeError: "sum_cpu" not implemented for 'Half' so must convert to float32.
        """
        layer_grad = [lg.type(torch.float32).cpu() for lg in layer_grad]
        avg_grads.append(torch.stack(layer_grad).mean(0).type(torch.float16))
    torch.save(avg_grads, state['transferDirectory'] + os.sep + out['avg_grads_file'])
    return out


def init_runs(cache, input):
    folds = []
    for site, site_vars in input.items():
        folds.append(list(zip(repeat(site), site_vars['splits'].items())))
    cache['folds'] = list(zip(*folds))
    cache.update(batch_size=[v['batch_size'] for _, v in input.items()][0])
    cache.update(id=[v['id'] for _, v in input.items()][0])


def next_run(cache, state):
    """
    This function pops a new fold, lock parameters, and forward init_nn signal to all sites
    """
    cache['fold'] = dict(cache['folds'].pop())
    seed = 244627
    log_dir = '_'.join([str(s) for s in set(f for _, (f, _) in cache['fold'].items())])
    cache.update(log_dir=state['outputDirectory'] + os.sep + cache['id'] + os.sep + log_dir)
    os.makedirs(cache['log_dir'], exist_ok=True)

    cache.update(best_val_score=0)
    cache.update(train_log=['Loss,Precision,Recall,F1,Accuracy'],
                 validation_log=['Loss,Precision,Recall,F1,Accuracy'],
                 test_log=['Loss,Precision,Recall,F1,Accuracy'])
    """
    **** Parameter Lock ******
    Batch sizes are distributed based on number of data items on each site.
    We lock batch size because small dataset sometimes leads to batch size of 1, 2 on a particular site. 
    It causes issue in batch norm computation. 
    So we make sure that all the parameters we pass on to sites do not compromise the training process.
    """
    train_lens = dict([(site, ln[1]) for site, ln in cache['fold'].items()])
    itr = sum(train_lens.values()) / cache['batch_size']
    batch_sizes = {}
    for site, bz in train_lens.items():
        batch_sizes[site] = int(max(bz // itr, 1) + 1)
    while sum(batch_sizes.values()) != cache['batch_size']:
        largest = max(batch_sizes, key=lambda k: batch_sizes[k])
        batch_sizes[largest] -= 1
    out = {}
    for site, (split_file, data_len) in cache['fold'].items():
        out[site] = {'split_ix': split_file, 'data_len': data_len,
                     'batch_size': batch_sizes[site], 'seed': seed}
    return out


def check(logic, k, v, input):
    phases = []
    for site_vars in input.values():
        phases.append(site_vars.get(k) == v)
    return logic(phases)


def on_epoch_end(cache, input):
    """
    #############################
    Entry status: "train_waiting"
    Exit status: "train"
    ############################

    This function runs once an epoch of training is done and all sites
        run the validation step i.e. all sites in "train_waiting" status.
    We accumulate training/validation loss and scores of the last epoch.
    We also send a save current model as best signal to all sites if the global validation score is better than the previously saved one.
    """
    out = {}
    train_prfa = Prf1a()
    train_loss = Avg()
    val_prfa = Prf1a()
    val_loss = Avg()
    for site, site_vars in input.items():
        for tl, tp in site_vars['train_log']:
            train_loss.add(tl['sum'], tl['count'])
            train_prfa.update(tp=tp['tp'], tn=tp['tn'], fn=tp['fn'], fp=tp['fp'])
        vl, vp = site_vars['validation_log']
        val_loss.add(vl['sum'], vl['count'])
        val_prfa.update(tp=vp['tp'], tn=vp['tn'], fn=vp['fn'], fp=vp['fp'])

    cache['train_log'].append([train_loss.average, *train_prfa.prfa()])
    cache['validation_log'].append([val_loss.average, *val_prfa.prfa()])

    if val_prfa.f1 >= cache['best_val_score']:
        cache['best_val_score'] = val_prfa.f1
        out['save_current_as_best'] = True
    else:
        out['save_current_as_best'] = False
    return out


def save_test_scores(cache, input):
    """
    ########################
    Entry: phase "next_run_waiting"
    Exit: continue on next fold if folds left, else success and stop
    #######################
    This function saves test score of last fold.
    """
    test_prfa = Prf1a()
    test_loss = Avg()
    for site, site_vars in input.items():
        vl, vp = site_vars['test_log']
        test_loss.add(vl['sum'], vl['count'])
        test_prfa.update(tp=vp['tp'], tn=vp['tn'], fn=vp['fn'], fp=vp['fp'])
    cache['test_log'].append([test_loss.average, *test_prfa.prfa()])
    cache['test_scores'] = json.dumps(vars(test_prfa))
    cache['global_test_score'].append(vars(test_prfa))
    core.utils.save_logs(cache, plot_keys=['train_log', 'validation_log'], file_keys=['test_log', 'test_scores'],
                         log_dir=cache['log_dir'])


def send_global_scores(cache, state):
    out = {}
    score = Prf1a()
    for sc in cache['global_test_score']:
        score.update(tp=sc['tp'], tn=sc['tn'], fn=sc['fn'], fp=sc['fp'])
    cache['global_test_score'] = ['Precision,Recall,F1,Accuracy']
    cache['global_test_score'].append(score.prfa())
    core.utils.save_logs(cache, file_keys=['global_test_score'],
                         log_dir=state['outputDirectory'] + os.sep + cache['id'])
    out['results_zip'] = f"{cache['id']}_" + '_'.join(str(datetime.datetime.now()).split(' '))
    shutil.make_archive(f"{state['transferDirectory']}{os.sep}{out['results_zip']}", 'zip', cache['log_dir'])
    return out


def set_mode(input, mode=None):
    out = {}
    for site, site_vars in input.items():
        out[site] = mode if mode else site_vars['mode']
    return out


if __name__ == "__main__":
    args = json.loads(sys.stdin.read())
    cache = args['cache']
    input = args['input']
    state = args['state']
    out = {}

    nxt_phase = input.get('phase', 'init_runs')
    if check(all, 'phase', 'init_runs', input):
        """
        Initialize all folds and loggers
        """
        cache['global_test_score'] = []
        init_runs(cache, input)
        out['run'] = next_run(cache, state)
        out['global_modes'] = set_mode(input)
        nxt_phase = 'init_nn'

    if check(all, 'phase', 'computation', input):
        """
        Main computation phase where we aggregate sites information
        We also handle train/validation/test stages of local sites by sending corresponding signals from here
        """
        nxt_phase = 'computation'
        if check(any, 'mode', 'train', input):
            out.update(**aggregate_sites_info(input))
            out['global_modes'] = set_mode(input)

        if check(all, 'mode', 'val_waiting', input):
            out['global_modes'] = set_mode(input, mode='validation')

        if check(all, 'mode', 'train_waiting', input):
            out.update(**on_epoch_end(cache, input))
            out['global_modes'] = set_mode(input, mode='train')

        if check(all, 'mode', 'test', input):
            out.update(**on_epoch_end(cache, input))
            out['global_modes'] = set_mode(input, mode='test')

    if check(all, 'phase', 'next_run_waiting', input):
        """
        This block runs when a fold has completed all train, test, validation phase.
        We save all the scores and plot the results.
        We transition to new fold if there is any left, else we stop the distributed computation with a success signal.
        """
        save_test_scores(cache, input)
        if len(cache['folds']) > 0:
            out['nn'] = {}
            out['run'] = next_run(cache, state)
            out['global_modes'] = set_mode(input)
            nxt_phase = 'init_nn'
        else:
            out.update(**send_global_scores(cache, state))
            nxt_phase = 'success'

    out['phase'] = nxt_phase
    output = json.dumps({'output': out, 'cache': cache, 'success': check(all, 'phase', 'success', input)})
    sys.stdout.write(output)
    args, cache, input, state, out, output = [None] * 6
