import torch
import torch.nn as nn


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.bias = bias
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

    def init_hidden(self, bz, device='cpu'):
        return (torch.zeros(bz, self.hidden_size, device=device),
                torch.zeros(bz, self.hidden_size, device=device))

    def forward(self, x, h=None):
        bsz, seq_sz, _ = x.shape
        HZ = self.hidden_size

        if h is None:
            h_t, c_t = self.init_hidden(bsz, x.device)
        else:
            h_t, c_t = h

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


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True, num_layers=1, bias=True):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_direction = 2 if bidirectional else 1
        self.hidden_size = hidden_size // self.num_direction
        self.bias = bias
        self.lstms = [LSTMCell(self.input_size, self.hidden_size) for _ in range(self.num_direction)]

    def forward(self, x, h=None):
        hidden_seq, (h_t, c_t) = self.lstms[0](x, h)
        if self.bidirectional:
            rev_hidden_seq, (rev_h_t, rev_c_t) = self.lstms[1](torch.flip(x, (1,)), h)
            hidden_seq = torch.cat([hidden_seq, rev_hidden_seq], 2)
            h_t = torch.cat([h_t, rev_h_t], 1)
            c_t = torch.cat([c_t, rev_c_t], 1)
        return hidden_seq, (h_t, c_t)


class ENcoder(nn.Module):

    def __init__(self,
                 input_size=20,
                 hidden_size=256,
                 bidirectional=True,
                 z=256,
                 seq_len=53):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1

        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional
        )

        self.features = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * seq_len, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, z),
            nn.ReLU(),
            nn.BatchNorm1d(z),
        )

    def forward(self, x):
        """Encode to low dim first"""
        o, h = self.lstm(x)
        return self.features(o.flatten(1)), h


class ICALstm(nn.Module):

    def __init__(self,
                 input_size=256,
                 hidden_size=256,
                 bidirectional=True,
                 num_cls=2,
                 num_comps=53,
                 window_size=20,
                 seq_len=7):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1

        self.num_comp = num_comps
        self.window_size = window_size

        # 32 * 13 * 53 * 20
        self.encoder = ENcoder(input_size=window_size, hidden_size=hidden_size // 2, seq_len=num_comps, z=input_size)
        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * seq_len, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_cls)
        )

    def forward(self, x):
        """Encode to low dim first"""
        x = torch.stack([self.encoder(b)[0] for b in x], 0)
        o, h = self.lstm(x)
        return self.classifier(o.flatten(1)), h

# i = torch.randn(13, 53, 20)
# m = ENcoder(input_size=20, seq_len=53)
# o = m(i)
# print(o[0].shape)
