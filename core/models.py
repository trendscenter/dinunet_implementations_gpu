import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class BasicConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, **kw):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kw)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class MXPConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, mxp_k=2, mxp_s=2, **kw):
        super(MXPConv3d, self).__init__()
        self.conv = BasicConv3d(in_channels, out_channels, **kw)
        self.mx_k = mxp_k
        self.mxp_s = mxp_s

    def forward(self, x):
        x = F.max_pool3d(x, kernel_size=self.mx_k, stride=self.mxp_s)
        return self.conv(x)


class UpConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kw):
        super(UpConv3d, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, **kw)
        self.conv = BasicConv3d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class VBMNet(nn.Module):
    def __init__(self, in_ch, num_class, r=2):
        super(VBMNet, self).__init__()
        self.c1 = BasicConv3d(in_ch, r, kernel_size=3)

        self.c2 = MXPConv3d(r, 2 * r, kernel_size=3)
        self.c3 = MXPConv3d(2 * r, 4 * r, kernel_size=3)
        self.c4 = MXPConv3d(4 * r, 8 * r, kernel_size=3)

        self.c5 = UpConv3d(8 * r, 4 * r, kernel_size=2, stride=2)
        self.c6 = UpConv3d(4 * r, 2 * r, kernel_size=2, stride=2)
        self.c7 = UpConv3d(2 * r, r, kernel_size=2, stride=2)

        self.c8 = MXPConv3d(2 * r, 4 * r, kernel_size=3)
        self.c9 = MXPConv3d(4 * r, 8 * r, kernel_size=3)
        self.c10 = MXPConv3d(8 * r, 16 * r, kernel_size=3)

        self.c11 = MXPConv3d(24 * r, r, kernel_size=1)

        self.flat_size = r * 3 * 5 * 3
        self.fc1 = nn.Linear(self.flat_size, 64)
        self.out = nn.Linear(64, num_class)

    def forward(self, x):
        x1 = self.c1(x)
        x = self.c2(x1)
        x = self.c3(x)
        x4 = self.c4(x)

        x = self.c5(x4)
        x = self.c6(x)
        x7 = self.c7(x)

        x = self.c8(self.crop_concat(x1, x7))
        x = self.c9(x)
        x10 = self.c10(x)

        x = self.c11(self.crop_concat(x4, x10))

        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1(x), inplace=True)
        return self.out(x)

    @staticmethod
    def crop_concat(large, small):
        diff = np.array(large.shape) - np.array(small.shape)
        diffa = np.floor(diff / 2).astype(int)
        diffb = np.ceil(diff / 2).astype(int)
        t = large[:, :, diffa[2]:large.shape[2] - diffb[2], diffa[3]:large.shape[3] - diffb[3],
            diffa[4]:large.shape[2] - diffb[4]]
        return torch.cat([t, small], 1)
#
#
# device = torch.device('cuda:0')
# m = VBMNet(1, 2, r=4)
# m = m.to(device)
#
# i = torch.randn((8, 1, 121, 145, 121))
# o = m(i.to(device))
# print(i.shape, o.shape)
#
# torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
# print('Total Params:', torch_total_params)
