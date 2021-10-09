import multiprocessing as mp
import time

import coinstac
from coinstac_dinunet import COINNRemote
from coinstac_dinunet.utils import duration

from comps import NNComputation, VBMTrainer

_cache = {}
_pool = None


def run(data):
    global _pool
    global _cache

    start_time = _cache.setdefault('start_time', time.time())
    if _pool is None:
        _pool = mp.Pool(processes=data['input'].get('num_reducers', 2))

    remote = COINNRemote(
        cache=_cache, input=data['input'], state=data['state']
    )

    """Add new NN computation Here"""
    if remote.cache['task_id'] == NNComputation.TASK_FREE_SURFER:
        args = VBMTrainer,
    else:
        raise ValueError(f"Invalid remote task:{remote.cache.get('task')}")

    out = remote(_pool, *args)

    _cache['total_remote_comp_duration'] = coinstac.compTime
    _cache['total_duration'] = f"{duration(start_time)}"

    return out
