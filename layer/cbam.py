import torch
import torch.nn as nn

'''
CBAM
desc:   Convolutional Block Attention Module
        channel & spatial dimension attention
Paper:  https://arxiv.org/abs/1807.06521
Code:   https://github.com/hujie-frank/SENet
Pytorch:https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
'''

class ChannelAttention(nn.Module):
    '''
    Channel dimensional attention
    improved from SENet channel attention

    Input:
        channel:   channel number of the input
        reduction: same as reduction in SENet

    Return:
        y which has same shape as x, but weighting in channel dimension
    '''
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            # replace Linear with 1x1 Conv
            nn.Conv2d(channel, channel // reduction, 1, bias=False), 
            nn.BatchNorm2d(channel // reduction),     # 后添加的
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.fc(self.avg_pool(x))
        maxout = self.fc(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    '''
    Spatial dimensional attention

    Input:
        kernel_size: size of visual receptive field for conv, must be 3 or 7

    Return:
        y which has same shape as x, weighting per pixel in HxW dimension
    '''
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    '''
    CBAM 分先后对 channel 和 spatial 进行注意力加权
    论文中对先后顺序进行了实验讨论，认为此顺序效果最佳

    实际使用中可自行处理
    '''
    def __init__(self, channel):
        super(CBAM,self).__init__()
        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()
    def forward(self, x):
       x = self.ca(x) * x  # 广播机制
       x = self.sa(x) * x  # 广播机制