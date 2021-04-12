# coding: utf-8
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbam import CBAM
from .resnet import ResNet, BasicBlock
from .rnn import GRU


class C3D_ResNet(nn.Module):
    def __init__(self, mode, inputDim=256, hiddenDim=512, nLayers=2, nClasses=500, frameLen=29, backend='tcn', every_frame=True, color_space='rgb', use_cbam=False):
        super(C3D_ResNet, self).__init__()
        self.mode = mode
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.nLayers = nLayers
        self.backend = backend
        self.every_frame = every_frame
        
        in_channels = 3 if color_space == 'rgb' else 1
        self.c3d = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], self.inputDim, zero_init_residual=True, use_cbam=use_cbam)
        
        if self.backend == 'gru':
            self.gru = GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses, self.every_frame)
        elif self.backend == 'tcn':
            self.backend_conv1 = nn.Sequential(
                nn.Conv1d(self.inputDim, 2*self.inputDim, 5, 2, 0, bias=False),
                nn.BatchNorm1d(2*self.inputDim),
                nn.ReLU(True),
                nn.MaxPool1d(2, 2),
                nn.Conv1d(2*self.inputDim, 4*self.inputDim, 5, 2, 0, bias=False),
                nn.BatchNorm1d(4*self.inputDim),
                nn.ReLU(True),
            )
            self.backend_conv2 = nn.Sequential(
                nn.Linear(4*self.inputDim, self.inputDim),
                nn.BatchNorm1d(self.inputDim),
                nn.ReLU(True),
                nn.Linear(self.inputDim, self.nClasses)
            )

        # initialize
        self._initialize_weights()

    def forward(self, x):
        b = x.size(0)
        x = self.c3d(x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        x = self.resnet(x)
        x = x.view(b, -1, self.inputDim)
        if self.backend == 'gru':
            x = self.gru(x)
        elif self.backend == 'tcn':
            x = x.transpose(1, 2)
            x = self.backend_conv1(x)
            x = torch.mean(x, 2)
            x = self.backend_conv2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
