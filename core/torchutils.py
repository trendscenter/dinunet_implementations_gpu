from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate


def safe_collate(batch):
    return default_collate([b for b in batch if b])


class NNDataLoader(DataLoader):

    def __init__(self, **kw):
        super(NNDataLoader, self).__init__(**kw)

    @classmethod
    def get_loader(cls, **kw):
        _kw = {
            'dataset': None,
            'batch_size': 1,
            'shuffle': False,
            'sampler': None,
            'batch_sampler': None,
            'num_workers': 0,
            'pin_memory': False,
            'drop_last': False,
            'timeout': 0,
            'worker_init_fn': None
        }
        for k in _kw.keys():
            _kw[k] = kw.get(k, _kw.get(k))
        return cls(collate_fn=safe_collate, **_kw)
