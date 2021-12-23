import torch
import torch.nn as nn


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
