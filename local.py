import multiprocessing as mp
import time

from coinstac_dinunet import COINNLocal
from coinstac_dinunet.utils import duration

from comps import AggEngine
from comps import FreeSurferTrainer, FreeSurferDataset, FSVDataHandle
from comps import ICATrainer, ICADataset, ICADataHandle
from comps import NNComputation, VBMTrainer, VBMDataset, VBMDataHandle

""" Test """
computation = NNComputation.TASK_FREESURFER
agg_engine = AggEngine.DECENTRALIZED_SGD

CACHE = {}
MP_POOL = None


def run(data):
    global CACHE
    global MP_POOL

    _start = time.time()
    start_time = CACHE.setdefault('start_time', _start)

    if MP_POOL is None and CACHE.get('num_reducers'):
        MP_POOL = mp.Pool(processes=CACHE['num_reducers'])

    dataloader_args = {"train": {"drop_last": True}}
    local = COINNLocal(
        task_id=computation, agg_engine=agg_engine,
        cache=CACHE, input=data['input'], batch_size=16,
        state=data['state'], epochs=101, patience=101, split_ratio=[0.8, 0.1, 0.1],
        pretrain_args=None, dataloader_args=dataloader_args
    )

    """Add new NN computation Here"""
    if local.cache['task_id'] == NNComputation.TASK_FREESURFER:
        args = FreeSurferTrainer, FreeSurferDataset, FSVDataHandle

    elif local.cache['task_id'] == NNComputation.TASK_VBM:
        args = VBMTrainer, VBMDataset, VBMDataHandle

    elif local.cache['task_id'] == NNComputation.TASK_ICA:
        args = ICATrainer, ICADataset, ICADataHandle

    else:
        raise ValueError(f"Invalid local task:{local.cache.get('task')}")

    out = local(MP_POOL, *args)

    duration(CACHE, _start, key='time_spent_on_computation')
    duration(CACHE, start_time, key='cumulative_total_duration')
    return out
