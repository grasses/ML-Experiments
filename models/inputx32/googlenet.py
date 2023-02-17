'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=43):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        self.num_classes = num_classes
        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def layerx1(self, x):
        return self.pre_layers(x)

    def layerx2(self, x):
        out = self.a3(x)
        out = self.b3(out)
        return self.maxpool(out)

    def layerx3(self, x):
        out = self.a4(x)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        return self.maxpool(out)

    def layerx4(self, x):
        out = self.a5(x)
        out = self.b5(out)
        return self.avgpool(out)


    def mid_forward(self, x, layer_index):
        """
        Feed x to model from $layer_index layer
        Args:
            self: Densenet
            x: Tensor
            layer_index: Int
        Returns: Tensor
        """
        if layer_index == 1:
            x = self.layerx2(x)
            x = self.layerx3(x)
            x = self.layerx4(x)
            x = self.layerx5(x)
        if layer_index == 2:
            x = self.layerx3(x)
            x = self.layerx4(x)
            x = self.layerx5(x)
        if layer_index == 3:
            x = self.layerx4(x)
            x = self.layerx5(x)
        if layer_index == 4:
            x = self.layerx5(x)
        out = x.view(x.size(0), -1)
        out = self.linear(out)
        return out

    def fed_forward(self, x, layer_index):
        """
        Feed x to model from head to $layer_index layer
        Args:
            self: Densenet
            x: Tensor
            layer_index: Int
        Returns: Tensor
        """
        x = self.layerx1(x)
        if layer_index == 1: return x.contiguous()
        x = self.layerx2(x)
        if layer_index == 2: return x.contiguous()
        x = self.layerx3(x)
        if layer_index == 3: return x.contiguous()
        x = self.layerx4(x)
        if layer_index == 4: return x.contiguous()
        x = self.layerx5(x)
        return x.contiguous()

    def feature_list(self, x, layer_index=[1, 2, 3, 4]):
        """
        Return feature map of each layer
        Args:
            self: Densenet
            x: Tensor
        Returns: Tensor, list
        """
        out_list = []
        x = self.layerx1(x)
        out_list.append(x.contiguous().view(x.size(0), -1).detach().cpu())
        x = self.layerx2(x)
        out_list.append(x.contiguous().view(x.size(0), -1).detach().cpu())
        x = self.layerx3(x)
        out_list.append(x.contiguous().view(x.size(0), -1).detach().cpu())
        x = self.layerx4(x)
        out_list.append(x.contiguous().view(x.size(0), -1).detach().cpu())
        out = x.view(x.size(0), -1)
        y = self.linear(out)
        return y, out_list


def googlenet(pretrained=False, progress=True, device="cpu", **kwargs):
    return GoogLeNet(**kwargs)


def test():
    net = GoogLeNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()