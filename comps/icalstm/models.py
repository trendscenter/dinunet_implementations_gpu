import torch
import torch.nn as nn


class _LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.bias = bias
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

    def forward(self, x, h):
        h_t, c_t = h
        bsz, seq_sz, _ = x.shape
        HZ = self.hidden_size

        hidden_seq = []
        for t in range(seq_sz):
            preact = self.i2h(x[:, t, :]) + self.h2h(h_t)
            gates = preact[:, :3 * self.hidden_size].sigmoid()
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HZ]),  # input gate
                torch.sigmoid(gates[:, HZ:HZ * 2]),  # forget gate
                torch.tanh(preact[:, HZ * 3:]),
                torch.sigmoid(gates[:, -HZ:]),  # output gate
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class ICALstm(nn.Module):

    def __init__(self,
                 input_size=256,
                 hidden_size=256,
                 num_cls=2,
                 num_comps=53,
                 window_size=20,
                 seq_len=7):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.direction = 1

        self.num_comp = num_comps
        self.window_size = window_size

        self.encoder = nn.Linear(self.num_comp * self.window_size, self.input_size)

        self.lstm = _LSTM(
            input_size=input_size,
            hidden_size=hidden_size
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.direction * hidden_size * seq_len, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_cls)
        )

        torch.manual_seed(1)

    def forward(self, x):
        """Encode to low dim first"""
        x = torch.stack([self.encoder(b.view(b.shape[0], -1)) for b in x])

        h = self.init_hidden(len(x))
        o, h = self.lstm(x, h)

        return self.classifier(o.flatten(1)), h

    def init_hidden(self, bz, device='cpu'):
        return (torch.zeros(self.direction * self.num_layers, bz, self.hidden_size, device=device).squeeze(),
                torch.zeros(self.direction * self.num_layers, bz, self.hidden_size, device=device).squeeze())
