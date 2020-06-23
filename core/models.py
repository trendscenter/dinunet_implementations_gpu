import torch.nn as nn
import torch.nn.functional as F
import torch


class CNNBlock(nn.Module):
    def __init__(self, in_size, out_size, k=3, s=1, p=0, d=1):
        super().__init__()
        layers = [nn.Conv3d(in_size, out_size, k, s, p, d),
                  nn.BatchNorm3d(out_size),
                  nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class VBMNet(nn.Module):
    def __init__(self, in_size, out_size):
        super(VBMNet, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.c1 = CNNBlock(in_size, 8, 3, 1)
        self.c2 = CNNBlock(8, 16, 3, 1, d=2)
        self.c3 = CNNBlock(16, 16, 3, 2)
        self.c4 = CNNBlock(16, 8, 1, 2)
        self.fc1 = nn.Linear(8 * 3 * 4 * 3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, out_size)

    def forward(self, x):
        x = F.max_pool3d(self.c1(x), 2)
        x = F.max_pool3d(self.c2(x), 2)
        x = F.max_pool3d(self.c3(x), 2)
        x = self.c4(x)
        x = F.relu(self.fc1(x.view(-1, 8 * 3 * 4 * 3)))
        x = F.relu(self.fc2(x))
        return self.out(x)