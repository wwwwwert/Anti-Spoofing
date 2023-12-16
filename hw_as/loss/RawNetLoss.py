import torch
from torch import Tensor, nn


class RawNetLoss(torch.nn.Module):
    def __init__(
            self
        ):
        super().__init__()
        weight = torch.FloatTensor([1.0, 9.0])
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def forward(
            self, 
            pred: torch.Tensor,
            audio_class: torch.Tensor,
            **batch
        ):
        loss = self.criterion(pred, audio_class)
        return loss
