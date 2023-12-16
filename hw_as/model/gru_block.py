import torch.nn as nn
import torch.nn.functional as F


class GRUBlock(nn.Module):
    def __init__(
            self, 
            in_dim: int, 
            h_dim: int=1024, 
            n_gru: int=3
        ):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=in_dim)
        self.lrelu = nn.LeakyReLU()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=h_dim,
            num_layers=n_gru,
            batch_first=True
        )
        
    def forward(self, x):
        x = self.bn(x)
        x = self.lrelu(x)
        x = x.transpose(2, 1)  # (batch, filt, time) -> (batch, time, filt)
        output, _ = self.gru(x)
        output = output[:, -1, :]
        return output
