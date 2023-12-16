from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gru_block import GRUBlock
from .res_block import ResBlocks
from .sinc_block import SincConv_fast


class RawNet(nn.Module):
    def __init__(
            self,
            first_resblock_channels: List,
            second_resblock_channels: List,
            gru_h_dim: int=1024,
            fc_h_dim: int=1024
        ) -> None:
        super().__init__()
        self.sinc_filters = SincConv_fast(out_channels=20, kernel_size=1024, min_band_hz=0, min_low_hz=0)  # поменять аргументы
        self.first_res_block = ResBlocks(first_resblock_channels)
        self.second_res_block = ResBlocks(second_resblock_channels)
        self.gru_block = GRUBlock(
            in_dim=second_resblock_channels[-1][-1],
            h_dim=gru_h_dim
        )
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=gru_h_dim,
			    out_features=fc_h_dim
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=fc_h_dim,
			    out_features=2
            )
        )

    def forward(self, audio: torch.Tensor, **batch):
        audio = audio.unsqueeze(1)
        output = self.sinc_filters(audio)
        output = F.max_pool1d(torch.abs(output), 3)

        output = self.first_res_block(output)
        output = self.second_res_block(output)
        output = self.gru_block(output)
        output = self.fc(output)

        return {'pred': output}
    
