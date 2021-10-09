import multiprocessing as mp
import time

import coinstac
from coinstac_dinunet import COINNLocal
from coinstac_dinunet.utils import duration

from comps import AggEngine
from comps import NNComputation, VBMTrainer, VBMDataset, VBMDataHandle

""" Test """
computation = NNComputation.TASK_FREE_SURFER
agg_engine = AggEngine.DECENTRALIZED_SGD

_cache = {}
_pool = None


def run(data):
    global _pool
    global _cache

    start_time = _cache.setdefault('start_time', time.time())
    if _pool is None:
        _pool = mp.Pool(processes=data['input'].get('num_reducers', 2))

    dataloader_args = {"train": {"drop_last": True}}
    local = COINNLocal(
        task_id=computation, agg_engine=agg_engine,
        cache=_cache, input=data['input'], batch_size=16,
        state=data['state'], epochs=15, patience=11, split_ratio=[0.8, 0.1, 0.1],
        pretrain_args=None, dataloader_args=dataloader_args
    )

    """Add new NN computation Here"""
    if local.cache['task_id'] == NNComputation.TASK_FREE_SURFER:
        args = VBMTrainer, VBMDataset, VBMDataHandle
    else:
        raise ValueError(f"Invalid local task:{local.cache.get('task')}")

    out = local(_pool, *args)
    _cache['total_local_comp_duration'] = coinstac.compTime
    _cache['total_duration'] = f"{duration(start_time)}"

    return out
