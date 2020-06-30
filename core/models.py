import torch.nn as nn
import torch.nn.functional as F
import torch


class BasicConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class MXPConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, mxp_k=2, mxp_s=2, **kwargs):
        super(MXPConv3d, self).__init__()
        self.conv = BasicConv3d(in_channels, out_channels, **kwargs)
        self.mx_k = mxp_k
        self.mxp_s = mxp_s

    def forward(self, x):
        x = self.conv(x)
        return F.max_pool3d(x, kernel_size=self.mx_k, stride=self.mxp_s)


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
    def __init__(self, in_ch, num_class):
        super(VBMNet, self).__init__()
        self.c1 = BasicConv3d(in_ch, 8, kernel_size=3)
        self.c2 = MXPConv3d(8, 16, kernel_size=3, dilation=2)
        self.c3 = MXPConv3d(16, 16, kernel_size=3, dilation=2)
        self.c4 = MXPConv3d(16, 8, kernel_size=3)
        self.c5 = MXPConv3d(8, 4, kernel_size=3)

        self.flat_size = 4 * 5 * 6 * 5
        self.fc1 = nn.Linear(self.flat_size, 64)
        self.out = nn.Linear(64, num_class)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)

        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1(x), inplace=True)
        return self.out(x)

# device = torch.device('cuda:0')
# m = VBMNet(1, 2)
# m = m.to(device)
# torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
# print('Total Params:', torch_total_params)
#
# i = torch.randn((8, 1, 121, 145, 121))
# o = m(i.to(device))
# print(i.shape)
