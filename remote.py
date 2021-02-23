#!/usr/bin/python
import sys
import json

from coinstac_dinunet import COINNRemote
from coinstac_dinunet.metrics import Prf1a


#
# import pydevd_pycharm
#
# pydevd_pycharm.settrace('172.17.0.1', port=8881, stdoutToServer=True, stderrToServer=True, suspend=False)


class NiftiRemote(COINNRemote):
    def _set_monitor_metric(self):
        self.cache['monitor_metric'] = 'f1', 'maximize'

    def _set_log_headers(self):
        self.cache['log_header'] = 'loss,accuracy,f1'

    def _new_metrics(self):
        return Prf1a()


if __name__ == "__main__":
    args = json.loads(sys.stdin.read())
    remote = NiftiRemote(cache=args['cache'], input=args['input'], state=args['state'])
    remote.compute()
    remote.send()
