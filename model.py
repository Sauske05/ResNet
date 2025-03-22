import torch

from torch import nn

'''
The Conv2d before maxpooling has Kernal of 7, padding of 3, stride of 2 -> (64,16,16)
The maxpooling has stride of 2 --> (64,8,8)

In each block, there are three layers. The first layer is used for skip connection
meaning the first layer has to have stride of 2 and padding of 0 to downsample. This is always the case
except for the first block's first layer. Inside the block, there is also skip connection, but it is not downsample,
but rather an identity skip connection. 



First Layer: -> First Block -> First Layer: Stride 1, Padding 0, Kernel 1 --> (64,8,8)
First Layer: -> First Block -> Second Layer: Stride 1, Padding 1, Kernel 3 --> (64,8,8)
First Layer: -> First Block -> Last Layer: Stride 1, Padding 0, Kernel 1 --> (256,8,8)
First Layer Identity Skip to Second Layer --> nn.Conv2d(64,256,1,0,1) --> (256,8,8)


Second Layer --> First Block --> First Layer input(256,8,8): Stride 2, kernel 1 and padding 0 -> (128,4,4)
Second Layer: -> First Block -> Second Layer: Stride 1, Padding 1, Kernel 3 --> (128,4,4)
Second Layer: -> First Block -> Last Layer: Stride 1, Padding 0, Kernel 1 --> (512,4,4)
Skip Connection --> Layer 1 to Layer 2 --> nn.Conv2d(256,512,1,2,0) padding is 0--> (512,4,4)


'''

import torch
import torch.nn as nn
from typing import List

class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1

        # Bottleneck layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)

        # Identity mapping
        self.identity = nn.Sequential()
        if in_channels != out_channels * 4 or downsample:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        identity = self.identity(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, architecture: List[int], num_classes: int = 10):
        super().__init__()
        
        # Initial convolutional layer (ResNet Stem)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Residual layers
        self.in_channels = 64
        self.layer1 = self._make_layer(64, architecture[0])
        self.layer2 = self._make_layer(128, architecture[1], downsample=True)
        self.layer3 = self._make_layer(256, architecture[2], downsample=True)
        self.layer4 = self._make_layer(512, architecture[3], downsample=True)

        # Fully connected layer (Classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, out_channels: int, blocks: int, downsample: bool = False):
        layers = []
        layers.append(Block(self.in_channels, out_channels, downsample))
        self.in_channels = out_channels * 4  # After first block, update input channels
        
        for _ in range(1, blocks):
            layers.append(Block(self.in_channels, out_channels, downsample=False))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



if __name__ == '__main__':

    resnet50 = ResNet50([3, 4, 6, 3])  
    image_sample = torch.rand(4,3,32,32)
    output = resnet50(image_sample)
    print(output.shape)
