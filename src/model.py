import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConv2d(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv2d(x)


class DownConv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            SimpleConv2d(in_ch, out_ch)
        )

    def forward(self, x):
        x, xs = x
        return self.down(x), (x, xs)


class UpConv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = SimpleConv2d(in_ch + out_ch, out_ch)

    def forward(self, x):
        x1, (x2, xs) = x
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x, xs


class BaselineModel(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.in_conv = SimpleConv2d(in_ch, 16)
        self.down = nn.Sequential(
            DownConv2d(16, 32),
            DownConv2d(32, 64),
            DownConv2d(64, 64),
        )
        self.up = nn.Sequential(
            UpConv2d(64, 64),
            UpConv2d(64, 32),
            UpConv2d(32, 16),
        )
        self.out_conv = nn.Conv2d(16, out_ch, 1)

    def forward(self, x):
        x = self.in_conv(x)
        x = x, None
        x = self.down(x)
        x = self.up(x)
        x, _ = x
        x = self.out_conv(x)
        return x


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)
