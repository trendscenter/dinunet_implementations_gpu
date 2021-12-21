import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = (20, 10)
import torch.nn as nn
import torch

hf = h5py.File("Data/COBRE_AllData.h5", "r")
raw_data = hf.get("COBRE_dataset")
raw_data = np.array(raw_data)
raw_data = raw_data.reshape(157, 100, 140)


def read_ix(file):
    return [int(float(l.strip())) for l in open(file).readlines()]


use_comps = read_ix('IndicesAndLabels/correct_indices_GSP.csv')
labels = read_ix('IndicesAndLabels/labels_COBRE.csv')

data = raw_data[:, use_comps, :]
labels = torch.from_numpy(np.array(labels) - 1)
_data = torch.from_numpy(data)

unfold = nn.Unfold(kernel_size=(1, 20), stride=(1, 5))
_data = unfold(_data.unsqueeze(2)).reshape(157, -1, 53, 20)
seq_len = _data.shape[1]


class ICALstm(nn.Module):

    def __init__(self,
                 input_size=256,
                 hidden_size=256,
                 num_layers=1,
                 num_cls=2,
                 bidirectional=True,
                 proj_size=64, num_comps=53,
                 window_size=20,
                 seq_len=7):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.direction = 2 if bidirectional else 1
        self.proj_size = proj_size

        self.num_comp = num_comps
        self.window_size = window_size

        self.encoder = nn.Linear(self.num_comp * self.window_size, self.input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            proj_size=proj_size
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.direction * proj_size * seq_len, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_cls)
        )

    def forward(self, x):
        """Encode to low dim first"""
        x = torch.stack([self.encoder(b.view(b.shape[0], -1)) for b in x])

        h = self.init_hidden(len(x))
        o, h = self.lstm(x, h)

        return self.classifier(o.flatten(1)), h

    def init_hidden(self, bz, device='cpu'):
        return (torch.zeros(self.direction * self.num_layers, bz, self.proj_size, device=device),
                torch.zeros(self.direction * self.num_layers, bz, self.hidden_size, device=device))
