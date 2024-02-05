# define resnet building blocks
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv2d, AdaptiveMaxPool2d 


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(Conv2d(inchannel, outchannel, kernel_size=3,
                                         stride=stride, padding=1, bias=False),
                                  nn.BatchNorm2d(outchannel),
                                  nn.ReLU(inplace=True),
                                  Conv2d(outchannel, outchannel, kernel_size=3,
                                         stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(outchannel))

        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:

            self.shortcut = nn.Sequential(Conv2d(inchannel, outchannel,
                                                 kernel_size=1, stride=stride,
                                                 padding = 0, bias=False),
                                          nn.BatchNorm2d(outchannel) )
    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BottleneckResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(BottleneckResidualBlock, self).__init__()
        
        # The bottleneck structure
        self.expansion = 4
        mid_channels = outchannel // self.expansion
        
        self.bottleneck = nn.Sequential(
            # 1x1 conv, reducing dimension
            Conv2d(inchannel, mid_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # 3x3 conv, the bottleneck layer
            Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # 1x1 conv, expanding dimension back to out_channels
            Conv2d(mid_channels, outchannel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:

            self.shortcut = nn.Sequential(Conv2d(inchannel, outchannel,
                                                 kernel_size=1, stride=stride,
                                                 padding = 0, bias=False),
                                          nn.BatchNorm2d(outchannel) )
    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class WideResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, widen_factor=2):
        super(WideResidualBlock, self).__init__()
        
        out_channels *= widen_factor
         
        self.main_path = nn.Sequential(
            # First convolutional layer
            Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Second convolutional layer
            Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Adjusting the shortcut path to match dimensions if needed
            self.shortcut = nn.Sequential(
                Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# define resnet

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 20):
        super(ResNet, self).__init__()
        self.inchannel = 16
        self.conv1 = nn.Sequential(Conv2d(3, 16, kernel_size = 3, stride = 1,
                                            padding = 1, bias = False),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU())
        channels = self.inchannel
        self.layers = nn.Sequential()
        self.widen_factor = 1 if block!=WideResidualBlock else 2
        for i, num_blocks in enumerate(layers):
            stride = 1 if i == 0 else 2  # First layer keeps size, others halve the feature map size
            layer = self.make_layer(block, channels, num_blocks, stride)
            self.layers.add_module(f'layer{i+1}', layer)
            channels *= 2 * self.widen_factor
            
        self.maxpool = AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(channels // 2 * self.widen_factor, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels * self.widen_factor
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.layers(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# please do not change the name of this class
def MyResNet(blocks, layers):
    if blocks=='residual':
        blocks = ResidualBlock
    elif blocks=='bottleneck':
        blocks = BottleneckResidualBlock
    elif blocks=='wide':
        blocks = WideResidualBlock
    else:
        raise ValueError(f"Unknown block type: {blocks}")
        
    return ResNet(blocks, layers)