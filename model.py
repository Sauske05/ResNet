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

class Block(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, identity_sample: bool = True, downsample:bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1,1,0)        
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 2,1,0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,1,1)
        self.conv3 = nn.Conv2d(out_channels, out_channels*4,1,1,0)
        
        self.identity = nn.Conv2d(in_channels, out_channels*4,1,2,0)
        if identity_sample:
            self.identity = nn.Conv2d(in_channels, out_channels*4, 1,0,1)
        
            

    def forward(self,x:torch.tensor):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x_identity = self.identity(x)
        print(f'Shape of x {x.shape}')
        print(f'Shape of x1 {x1.shape}')
        print(f'Shape of x2 {x2.shape}')
        print(f'Shape of x3 {x3.shape}')
        print(f'Shape of x_identity {x_identity.shape}')
        return x3+ x_identity
    
from typing import List
class ResNet50(nn.Module):
    def __init__(self, architecture:List[int]): #[3,4,6,3]
        super().__init__()
        for external_layer in architecture:
            for x in range(1, external_layer+1):
                downsample = True if external_layer > 1 and x == 1 else False
                f'block_{x}' = Block()
                # if external_layer == 1 and blocks == 1:
                #     downsample
    def forward(self,x):
        pass