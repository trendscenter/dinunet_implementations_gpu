import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class _DoubleConvolution(nn.Module):
    def __init__(self, in_channels, middle_channel, out_channels, p=1, k=3):
        super(_DoubleConvolution, self).__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=k, padding=p),
            nn.BatchNorm3d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True)
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class VBMNet(nn.Module):
    def __init__(self, num_channels, num_classes, reduce_by=1):
        super(VBMNet, self).__init__()
        self.A1_ = _DoubleConvolution(num_channels, int(64 / reduce_by), int(64 / reduce_by))
        self.A2_ = _DoubleConvolution(int(64 / reduce_by), int(128 / reduce_by), int(128 / reduce_by))
        self.A3_ = _DoubleConvolution(int(128 / reduce_by), int(256 / reduce_by), int(256 / reduce_by))

        self.A_mid = _DoubleConvolution(int(256 / reduce_by), int(512 / reduce_by), int(512 / reduce_by))

        self.A3_up = nn.ConvTranspose3d(int(512 / reduce_by), int(256 / reduce_by), kernel_size=2, stride=2)
        self._A3 = _DoubleConvolution(int(512 / reduce_by), int(256 / reduce_by), int(256 / reduce_by))

        self.A2_up = nn.ConvTranspose3d(int(256 / reduce_by), int(128 / reduce_by), kernel_size=2, stride=2)
        self._A2 = _DoubleConvolution(int(256 / reduce_by), int(128 / reduce_by), int(128 / reduce_by))

        self.A1_up = nn.ConvTranspose3d(int(128 / reduce_by), int(64 / reduce_by), kernel_size=2, stride=2)
        self._A1 = _DoubleConvolution(int(128 / reduce_by), int(64 / reduce_by), int(64 / reduce_by))

        self.enc1 = _DoubleConvolution(int(64 / reduce_by), int(64 / reduce_by), int(128 / reduce_by), p=0)
        self.enc2 = _DoubleConvolution(int(128 / reduce_by), int(128 / reduce_by), int(256 / reduce_by), p=0)
        self.enc3 = _DoubleConvolution(int(256 / reduce_by), int(256 / reduce_by), int(128 / reduce_by), p=0)
        self.enc4 = _DoubleConvolution(int(128 / reduce_by), int(128 / reduce_by), int(64 / reduce_by), p=0)

        self.flat_size = int(64 / reduce_by) * 3 * 5 * 3
        self.fc1 = nn.Linear(self.flat_size, 256)
        self.fc1_bn = nn.BatchNorm1d(256, track_running_stats=False)

        self.fc2 = nn.Linear(256, 128)
        self.fc2_bn = nn.BatchNorm1d(128, track_running_stats=False)

        self.fc3 = nn.Linear(128, 32)
        self.fc3_bn = nn.BatchNorm1d(32, track_running_stats=False)

        self.out = nn.Linear(32, num_classes)

    def forward(self, x):
        a1_ = self.A1_(x)
        a1_dwn = F.max_pool3d(a1_, kernel_size=2, stride=2)

        a2_ = self.A2_(a1_dwn)
        a2_dwn = F.max_pool3d(a2_, kernel_size=2, stride=2)

        a3_ = self.A3_(a2_dwn)
        a3_dwn = F.max_pool3d(a3_, kernel_size=2, stride=2)

        a_mid = self.A_mid(a3_dwn)

        a3_up = self.A3_up(a_mid)
        _a3 = self._A3(VBMNet.crop_concat(a3_, a3_up))

        a2_up = self.A2_up(_a3)
        _a2 = self._A2(VBMNet.crop_concat(a2_, a2_up))

        a1_up = self.A1_up(_a2)
        _a1 = self._A1(VBMNet.crop_concat(a1_, a1_up))

        _a1 = F.max_pool3d(_a1, kernel_size=2, stride=2)
        _a1 = self.enc1(_a1)

        _a1 = F.max_pool3d(_a1, kernel_size=2, stride=2)
        _a1 = self.enc2(_a1)

        _a1 = F.max_pool3d(_a1, kernel_size=2, stride=2)
        _a1 = self.enc3(_a1)

        _a1 = F.max_pool3d(_a1, kernel_size=2, stride=2)
        _a1 = self.enc4(_a1)

        _a1 = _a1.view(-1, self.flat_size)
        _a1 = F.relu(self.fc1_bn(self.fc1(_a1)))
        _a1 = F.relu(self.fc2_bn(self.fc2(_a1)))
        _a1 = F.relu(self.fc3_bn(self.fc3(_a1)))
        _a1 = self.out(_a1)
        return _a1

    @staticmethod
    def crop_concat(large, small):
        diff = np.array(large.shape) - np.array(small.shape)
        diffa = np.floor(diff / 2).astype(int)
        diffb = np.ceil(diff / 2).astype(int)
        t = large[:, :, diffa[2]:large.shape[2] - diffb[2], diffa[3]:large.shape[3] - diffb[3],
            diffa[4]:large.shape[2] - diffb[4]]
        return torch.cat([t, small], 1)

# #
# m = VBMNet(1, 2, 8)
# print(m)
# params_count = sum(p.numel() for p in m.parameters() if p.requires_grad)
# print(params_count)
# i = torch.randn((2, 1, 121, 145, 121))
# o = m(i)
# print("Out shape:", o.shape)
