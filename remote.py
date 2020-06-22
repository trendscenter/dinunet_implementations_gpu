#!/usr/bin/python
import json
import os
import sys
from itertools import repeat

import numpy as np

import core.utils as utils
from core.measurements import Prf1a, Avg


# pydevd_pycharm.settrace('172.17.0.1', port=8881, stdoutToServer=True, stderrToServer=True, suspend=False)


def aggregate_sites_grad(input):
    out = {}
    grads = []
    for site, site_vars in input.items():
        grad_file = state['baseDirectory'] + os.sep + site + os.sep + site_vars['grads_file']
        grad = np.load(grad_file, allow_pickle=True)
        grads.append(grad)
    out['avg_grads_file'] = 'avg_grads.npy'
    avg_grads = []
    for layer_grad in zip(*grads):
        avg_grads.append(np.array(layer_grad).mean(0))
    np.save(state['transferDirectory'] + os.sep + out['avg_grads_file'], np.array(avg_grads))
    return out


def init_nn_params(cache):
    out = {}
    out['input_size'] = 66
    out['hidden_sizes'] = [16, 8, 4, 2]
    out['num_class'] = 2
    out['epochs'] = 21
    out['learning_rate'] = 0.001
    cache['batch_size'] = 32
    cache['eid'] = 'volumetric'
    out['eid'] = 'volumetric'
    cache['global_test_score'] = []
    return out


def generate_folds(cache, input):
    folds = []
    for site, site_vars in input.items():
        folds.append(list(zip(repeat(site), site_vars['splits'].items())))
    cache['folds'] = list(zip(*folds))


def next_run(cache, state):
    seed = 244627

    fold = dict(cache['folds'].pop())
    log_dir = '_'.join([str(s) for s in set(f for _, (f, _) in fold.items())])
    cache.update(log_dir=state['outputDirectory'] + os.sep + cache['eid'] + os.sep + log_dir)
    os.makedirs(cache['log_dir'], exist_ok=True)

    cache.update(best_val_score=0)
    cache.update(train_log=['Loss,Precision,Recall,F1,Accuracy'],
                 validation_log=['Loss,Precision,Recall,F1,Accuracy'],
                 test_log=['Loss,Precision,Recall,F1,Accuracy'])

    train_lens = dict([(site, ln[1]) for site, ln in fold.items()])
    itr = sum(train_lens.values()) / cache['batch_size']
    batch_sizes = {}
    for site, bz in train_lens.items():
        batch_sizes[site] = int(max(bz // itr, 1) + 1)
    while sum(batch_sizes.values()) != cache['batch_size']:
        largest = max(batch_sizes, key=lambda k: batch_sizes[k])
        batch_sizes[largest] -= 1
    out = {}
    for site, (split_file, data_len) in fold.items():
        out[site] = {'split_ix': split_file, 'data_len': data_len,
                     'batch_size': batch_sizes[site], 'seed': seed}
    return out


def check(logic, k, v, input):
    phases = []
    for site_vars in input.values():
        phases.append(site_vars.get(k) == v)
    return logic(phases)


def on_epoch_end(cache, input):
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
    test_prfa = Prf1a()
    test_loss = Avg()
    for site, site_vars in input.items():
        vl, vp = site_vars['test_log']
        test_loss.add(vl['sum'], vl['count'])
        test_prfa.update(tp=vp['tp'], tn=vp['tn'], fn=vp['fn'], fp=vp['fp'])
    cache['test_log'].append([test_loss.average, *test_prfa.prfa()])
    cache['test_scores'] = json.dumps(vars(test_prfa))
    cache['global_test_score'].append(vars(test_prfa))
    utils.save_logs(cache, plot_keys=['train_log', 'validation_log'], file_keys=['test_log', 'test_scores'],
                    log_dir=cache['log_dir'])


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
        out['nn'] = init_nn_params(cache)
        generate_folds(cache, input)
        out['run'] = next_run(cache, state)
        out['global_modes'] = set_mode(input)
        nxt_phase = 'init_nn'

    if check(all, 'phase', 'computation', input):
        nxt_phase = 'computation'
        if check(any, 'mode', 'train', input):
            out.update(**aggregate_sites_grad(input))
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
        save_test_scores(cache, input)
        if len(cache['folds']) > 0:
            out['nn'] = {}
            out['run'] = next_run(cache, state)
            out['global_modes'] = set_mode(input)
            nxt_phase = 'init_nn'
        else:
            score = Prf1a()
            for sc in cache['global_test_score']:
                score.update(tp=sc['tp'], tn=sc['tn'], fn=sc['fn'], fp=sc['fp'])
            cache['global_test_score'] = ['Precision,Recall,F1,Accuracy']
            cache['global_test_score'].append(score.prfa())
            utils.save_logs(cache, file_keys=['global_test_score'],
                            log_dir=state['outputDirectory'] + os.sep + cache['eid'])
            nxt_phase = 'success'

    out['phase'] = nxt_phase
    output = json.dumps({'output': out, 'cache': cache, 'success': check(all, 'phase', 'success', input)})
    sys.stdout.write(output)
    args, cache, input, state, out, output = [None] * 6
