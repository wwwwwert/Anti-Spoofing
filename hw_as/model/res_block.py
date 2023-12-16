from typing import List

import torch
import torch.nn as nn


class ResBlocks(nn.Module):
    def __init__(
            self, 
            n_channels: List, 
        ):
        super().__init__()
        self.layers = nn.Sequential(*[
            ResBlock(in_channels, out_channels)
            for in_channels, out_channels in n_channels
        ])
    
    def forward(self, x):
        output = self.layers(x)
        return output


class ResBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int, 
        ):
        super().__init__()
        
        self.bn_1 = nn.BatchNorm1d(num_features=in_channels)
        self.lr_1 = nn.LeakyReLU(negative_slope=0.3)
        
        self.conv_1 = nn.Conv1d(
            in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=3,
			padding=1,
			stride=1
        )
        
        self.bn_2 = nn.BatchNorm1d(num_features=out_channels)
        self.lr_2 = nn.LeakyReLU(negative_slope=0.3)

        self.conv_2 = nn.Conv1d(in_channels=out_channels,
			out_channels=out_channels,
			padding=1,
			kernel_size=3,
			stride=1
        )
        
        self.downsample = (in_channels != out_channels)
        if self.downsample:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(
                in_channels=in_channels,
				out_channels=out_channels,
				padding=0,
				kernel_size=1,
				stride=1
            )

        self.pooling = nn.MaxPool1d(3)
        self.fms = FMSBlock(out_channels)

    def forward(self, x: torch.Tensor):
        residual = x

        output = self.bn_1(x)
        output = self.lr_1(output)
        output = self.conv_1(output)

        output = self.bn_2(output)
        output = self.lr_2(output)
        output = self.conv_2(output)
        
        if self.downsample:
            residual = self.conv_downsample(residual)
        
        output += residual
        output = self.pooling(output)
        output = self.fms(output)
        return output


class FMSBlock(nn.Module):
    def __init__(
            self,
            n_channels: int,
        ):
        super().__init__()
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc_attention = nn.Linear(
            in_features=n_channels,
            out_features=n_channels
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        y = self.avg_pooling(x).view(x.size(0), -1)
        y = self.fc_attention(y)
        y = self.sigmoid(y).view(y.size(0), y.size(1), -1)
        x = x * y + y
        return x
