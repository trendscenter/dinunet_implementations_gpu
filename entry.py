#!/usr/bin/python

from coinstac_dinunet import COINNLocal, COINNRemote
from coinstac_dinunet.io import COINPyService
from coinstac_dinunet.metrics import Prf1a

from classification import NiftiDataset, NiftiTrainer


class VBMRemote(COINNRemote):
    def _set_monitor_metric(self):
        self.cache['monitor_metric'] = 'f1', 'maximize'

    def _set_log_headers(self):
        self.cache['log_header'] = 'Loss|Accuracy,F1'

    def _new_metrics(self):
        return Prf1a()


class Server(COINPyService):

    def get_local(self, msg) -> callable:
        pretrain_args = {'epochs': 51, 'batch_size': 16}
        local = COINNLocal(cache=self.cache, input=msg['data']['input'],
                           pretrain_args=None, batch_size=8, model_scale=4,
                           state=msg['data']['state'], epochs=21, patience=21, computation_id='vbm_quick')
        return local

    def get_remote(self, msg) -> callable:
        remote = VBMRemote(cache=self.cache, input=msg['data']['input'],
                           state=msg['data']['state'])
        return remote

    def get_local_compute_args(self, msg) -> list:
        return [NiftiTrainer, NiftiDataset]


server = Server(verbose=False)
server.start()
