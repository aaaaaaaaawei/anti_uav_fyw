from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F


__all__ = ['SiamFC']


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001): #out_scale输出特征图的缩放比例
        super(SiamFC, self).__init__()
        self.out_scale = out_scale
    
    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale
    
    def _fast_xcorr(self, z, x):
        # fast cross correlation 快速的交叉相关运算
        nz = z.size(0)#获取z张量的批大小，即z张量在批次维度上的大小
        nx, c, h, w = x.size()#获取x张量的形状信息，包括批大小(nx)、通道数(c)、高度(h)和宽度(w)。
        x = x.view(-1, nz * c, h, w)#将x张量的形状变换为(-1, nz * c, h, w)，nz*c表示将x张量的批次和通道维度合并为一个维度，以便与z张量进行卷积。
        out = F.conv2d(x, z, groups=nz) #x是输入张量，z是卷积核，groups=nz表示将x张量拆分成nz组，然后每组与z张量进行卷积
        out = out.view(nx, -1, out.size(-2), out.size(-1))#将卷积后输出张量变形为正确的形状，其中nx表示批次维度大小，-1表示自动计算该维度的大小，out.size(-2)和out.size(-1)表示高度和宽度维度的大小。
        return out
