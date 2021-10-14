import os

import pandas as pd
import torch
import torch.nn.functional as F
from coinstac_dinunet import COINNDataset, COINNTrainer, COINNDataHandle
from coinstac_dinunet.data.datautils import init_k_folds
from coinstac_dinunet.metrics import Prf1a

from .models import MSANNet


class FreeSurferDataset(COINNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.labels = {}

    def load_index(self, site, file):
        if self.labels.get(site) is None:
            self.labels[site] = pd.DataFrame(self.inputspecs[site]['covariates']).T
        y = self.labels[site].loc[file][self.inputspecs[site]['labels_column']]

        if isinstance(y, str):
            y = int(y.strip().lower() == 'true')

        """
        int64 could not be json serializable.
        """
        self.indices.append([site, file, int(y)])

    def __getitem__(self, ix):
        site, file, y = self.indices[ix]
        data_dir = self.path(site)
        df = pd.read_csv(data_dir + os.sep + file, sep='\t', names=['File', file], skiprows=1)
        df = df.set_index(df.columns[0])
        df = df / df.max().astype('float64')
        x = df.T.iloc[0].values
        return {'inputs': torch.tensor(x), 'labels': torch.tensor(y), 'ix': torch.tensor(ix)}


class FreeSurferTrainer(COINNTrainer):
    def __init__(self, **kw):
        super().__init__(**kw)

    def _init_nn_model(self):
        self.nn['fs_net'] = MSANNet(in_size=self.cache['input_size'],
                                    hidden_sizes=self.cache['hidden_sizes'], out_size=self.cache['num_class'])

    def iteration(self, batch):
        inputs, labels = batch['inputs'].to(self.device['gpu']).float(), batch['labels'].to(self.device['gpu']).long()
        indices = batch['ix'].to(self.device['gpu']).long()

        out = F.log_softmax(self.nn['fs_net'](inputs), 1)
        wt = torch.randint(1, 101, (2,)).to(self.device['gpu']).float()
        loss = F.nll_loss(out, labels, weight=wt)

        _, predicted = torch.max(out, 1)
        score = self.new_metrics()
        score.add(predicted, labels)
        val = self.new_averages()
        val.add(loss.item(), len(inputs))
        return {'out': out, 'loss': loss, 'averages': val, 'metrics': score, 'prediction': predicted,
                'indices': indices}

    def _set_monitor_metric(self):
        self.cache['monitor_metric'] = 'f1', 'maximize'

    def _set_log_headers(self):
        self.cache['log_header'] = 'Loss|Accuracy,F1'

    def new_metrics(self):
        return Prf1a()


class FSVDataHandle(COINNDataHandle):
    def prepare_data(self):
        files = list(pd.DataFrame(self.cache['covariates']).T.index)
        return init_k_folds(files, self.cache, self.state)
