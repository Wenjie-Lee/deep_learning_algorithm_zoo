import torch
import torch.nn as nn

'''
SENet
desc:   Squeeze-and-Excitation Networks
Paper:  https://arxiv.org/abs/1709.01507
Code:   https://github.com/hujie-frank/SENet
Pytorch:https://github.com/moskomule/senet.pytorch 
'''

class SELayer(nn.Module):
    '''
    Squeeze-and-Excitation Networks

    Input:
        channel:    channel number of the input
        reduction:  bottleneck reduce scale for linear layer
    
    Return:
        y which has the same shape as x, but weighting in channel dimension
                                         for channel attention
    '''
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # nn.Linear(channel, channel // reduction, bias=False),
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.Dropout2d(p=0.1),        # 模拟Dropout, 丢失任意整个channel, 原文没有这一步
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel, bias=False),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)