import numpy as np
import torch

from hw_as.base.base_metric import BaseMetric

from .calculate_eer import compute_eer


class EERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.softmax = torch.nn.Softmax(dim=-1)

    def __call__(self, pred: torch.Tensor, audio_class: torch.Tensor, **batch):
        pred_probas = self.softmax(pred)
        target_scores = pred_probas[:, 1]
        eer, thr = compute_eer(
            bonafide_scores=target_scores[audio_class == 1].cpu().detach().numpy(),
            other_scores=target_scores[audio_class == 0].cpu().detach().numpy()
        )

        return eer
    
    @staticmethod
    def calc_err(original_probas: np.array, targets: np.array):
        eer, thr = compute_eer(
            bonafide_scores=original_probas[targets == 1],
            other_scores=original_probas[targets == 0]
        )

        return eer