import multiprocessing as mp
import time

from coinstac_dinunet import COINNRemote
from coinstac_dinunet.utils import duration

from comps import NNComputation, VBMTrainer, FreeSurferTrainer

CACHE = {}
MP_POOL = None


def run(data):
    global CACHE
    global MP_POOL

    _start = time.time()
    start_time = CACHE.setdefault('start_time', _start)

    if MP_POOL is None:
        MP_POOL = mp.Pool(processes=data['input'].get('num_reducers', 2))

    remote = COINNRemote(
        cache=CACHE, input=data['input'], state=data['state']
    )

    """Add new NN computation Here"""
    if remote.cache['task_id'] == NNComputation.TASK_FREESURFER:
        args = FreeSurferTrainer,

    if remote.cache['task_id'] == NNComputation.TASK_VBM:
        args = VBMTrainer,
    else:
        raise ValueError(f"Invalid remote task:{remote.cache.get('task')}")

    out = remote(MP_POOL, *args)

    duration(CACHE, _start, key='time_spent_on_computation')
    duration(CACHE, start_time, key='total_duration')
    return out
