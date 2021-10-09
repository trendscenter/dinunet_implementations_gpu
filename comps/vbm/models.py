import torch
from torch import nn


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, p=1, k=3):
        super(DownBlock, self).__init__()
        self.layers = nn.ModuleList()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=k, padding=p),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class VBMNet(nn.Module):
    def __init__(self, num_channels, num_classes, b_mul=1):
        super().__init__()
        self.features = nn.Sequential(
            DownBlock(num_channels, int(4 * b_mul)),
            DownBlock(int(4 * b_mul), int(8 * b_mul)),
            DownBlock(int(8 * b_mul), int(16 * b_mul)),
            DownBlock(int(16 * b_mul), int(24 * b_mul)),
            DownBlock(int(24 * b_mul), int(32 * b_mul)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(int(32 * b_mul) * 3 * 4 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

#
# m = VBMNet(1, 2, b_mul=2)
# print(m.classifier.training)
# m.eval()
# print(m.classifier.training)
# i = torch.randn((1, 1, 121, 145, 121))
# o = m(i)
# print("Out shape:", o.shape)
