import torch
from torch import nn
from torch.nn import functional as F

class WeightNormConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias))
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        return self.bn(self.conv(x))



class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = WeightNormConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = WeightNormConv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.downsample = downsample

        self.conv1.conv.weight.data.normal_(0, 0.1)
        self.conv1.conv.bias.data.zero_()
        self.conv2.conv.weight.data.normal_(0, 0.1)
        self.conv2.conv.bias.data.zero_()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # project if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




class BottleneckBlock(nn.Module):
  expansion: int = 4
  def __init__(self, in_channels, out_channels, downsample = None):
    
    super().__init__()
    layers =  [
      WeightNormConv2d(in_channels, out_channels, kernel_size=1),
      nn.ReLU(),
      WeightNormConv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.ReLU(),
      WeightNormConv2d(out_channels, out_channels * self.expansion, kernel_size=1),
    ]
    self.net = nn.Sequential(*layers)

    self.net[0].conv.weight.data.normal_(0, 0.05)
    self.net[0].conv.bias.data.zero_()

    self.net[-1].conv.weight.data.normal_(0, 0.05)
    self.net[-1].conv.bias.data.zero_()

    self.downsample = downsample

  def forward(self, x):
    out = self.net(x)
    if self.downsample is not None:
      x = self.downsample(x) 
    out = F.relu(out + x)
    return out
