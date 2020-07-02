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
        x = self.conv(x)
        return F.max_pool3d(x, kernel_size=self.mx_k, stride=self.mxp_s)


class UpConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kw):
        super(UpConv3d, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, **kw)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, conv_block, scale=16):
        super(InceptionA, self).__init__()
        self.branch1x1 = conv_block(in_channels, 2 * scale, kernel_size=1)
        self.branch3x3dbl_1 = conv_block(in_channels, 2 * scale, kernel_size=3, padding=1)
        self.branch3x3dbl_2 = conv_block(2 * scale, 3 * scale, kernel_size=3, padding=1)
        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch_pool = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class VBMNet(nn.Module):
    def __init__(self, in_ch, num_class, r=4):
        super(VBMNet, self).__init__()
        self.c1 = MXPConv3d(in_ch, r, kernel_size=3)

        self.c2 = MXPConv3d(r, 2 * r, kernel_size=3)
        self.c3 = MXPConv3d(2 * r, 4 * r, kernel_size=3)

        self.c4 = UpConv3d(4 * r, 2 * r, kernel_size=2, stride=2)
        self.c5 = UpConv3d(2 * r, r, kernel_size=2, stride=2)

        self.c6 = MXPConv3d(2 * r, 4 * r, kernel_size=3)
        self.c7 = MXPConv3d(4 * r, 2 * r, kernel_size=3)

        self.cat = MXPConv3d(6 * r, 4, kernel_size=3)

        # self.drop = nn.Dropout3d(p=0.5)
        self.flat_size = 4 * 4 * 6 * 4
        self.fc1 = nn.Linear(self.flat_size, 64)
        self.out = nn.Linear(64, num_class)

    def forward(self, x):
        x1 = self.c1(x)

        x = self.c2(x1)
        x3 = self.c3(x)

        x = self.c4(x3)
        x5 = self.c5(x)

        x = self.c6(self.crop_concat(x1, x5))
        x7 = self.c7(x)

        x = self.cat(self.crop_concat(x3, x7))

        # x = self.drop(x)
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
# device = torch.device('cuda:0')
# m = VBMNet(1, 2, r=16)
# m = m.to(device)
#
# i = torch.randn((8, 1, 121, 145, 121))
# o = m(i.to(device))
# print(i.shape)

# torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
# print('Total Params:', torch_total_params)
