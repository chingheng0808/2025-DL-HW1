import torch
import torch.nn.functional as F
import torch.nn as nn


class Model2(nn.Module):
    def __init__(self, in_channels=3, dim=256, num_classes=50, dropout_rate=0.2):
        super(Model2, self).__init__()
        self.dim = dim
        self.large_kernel = nn.Parameter(
            torch.randn(dim // 4, dim // 4, 9, 9), requires_grad=True
        )

        self.conv_first = nn.Conv2d(
            in_channels, dim, kernel_size=5, stride=2, padding=2
        )  # mapping to dim
        self.bn1 = nn.BatchNorm2d(dim)

        self.wide_conv1 = WideConv(dim, self.large_kernel)
        self.wide_conv2 = WideConv(dim, self.large_kernel)

        ## projection
        self.fc = nn.Linear(dim, num_classes, bias=True)
        self.fc.bias.data.zero_()

        self.relu = nn.ReLU(inplace=True)
        self.adv_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        ## layer 1
        x = self.conv_first(x)  # (B, dim, 128, 128)
        x = self.bn1(x)
        x = self.relu(x)  # (B, dim, 128, 128)

        ## layer 2
        x = self.wide_conv1(x)  # (B, dim, 128, 128)
        x = 0.5 * F.max_pool2d(
            x,
            kernel_size=(2, 2),
            stride=2,
        ) + 0.5 * F.avg_pool2d(
            x, kernel_size=(2, 2), stride=2
        )  # (B, dim, 64, 64)

        ## layer 3
        x = self.wide_conv2(x)  # (B, dim, 64, 64)
        x = self.adv_pool(x)  # (B, dim, 1, 1)

        ## flatten
        x = torch.flatten(x, 1)  # (B, dim)
        # x = self.dropout(x)
        x = self.fc(x)  # (B, num_classes)

        return x


class WideConv(nn.Module):
    def __init__(self, dim, large_kernel):
        super(WideConv, self).__init__()
        self.large_kernel = large_kernel
        ## different scales of convs
        self.conv_1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.conv_a = nn.Conv2d(
            dim * 3 // 4,
            dim * 3 // 4,
            kernel_size=3,
            dilation=1,
            stride=1,
            padding=1,
            groups=dim * 3 // 16,
        )
        self.conv_b = nn.Conv2d(
            dim * 3 // 4,
            dim * 3 // 4,
            kernel_size=3,
            dilation=2,
            stride=1,
            padding=2,
            groups=dim * 3 // 16,
        )
        self.conv_c = nn.Conv2d(
            dim * 3 // 4,
            dim * 3 // 4,
            kernel_size=3,
            dilation=3,
            stride=1,
            padding=3,
            groups=dim * 3 // 16,
        )
        self.w = nn.Parameter(torch.ones(4), requires_grad=True)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(dim)

        ## SE module
        self.se = SE(dim, reduction=16)

    def forward(self, x):
        a = self.forward_lk(x, self.large_kernel, self.conv_a)
        b = self.forward_lk(x, self.large_kernel, self.conv_b)
        c = self.forward_lk(x, self.large_kernel, self.conv_c)
        w_norm = torch.softmax(self.w, dim=0)
        x = (
            w_norm[0] * self.conv_1(x) + w_norm[1] * a + w_norm[2] * b + w_norm[3] * c
        )  # (B, dim, 128, 128)
        x = self.bn(x)
        x = self.relu(x)  # (B, dim, 128, 128)
        x = self.se(x)
        x = self.bn2(x)
        return x

    def forward_lk(self, x, large_kernel, conv):
        B, C, H, W = x.size()
        x1 = F.conv2d(
            x[:, : C // 4, :, :],
            large_kernel,
            stride=1,
            padding=large_kernel.shape[-1] // 2,
        )
        x2 = conv(x[:, C // 4 :, :, :])
        x = torch.cat([x1, x2], dim=1)  # (B, dim*3/4 + dim*1/4, 128, 128)
        return x


class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w
