import os
import torch
import torch.nn as nn
from torch import Tensor


class Losser:
    """
        Loss class for network training, you can add your loss in here
    """

    def __init__(self):
        self.wm_criterion = WeightMapLoss()

    def get_loss(self, output: Tensor, label: Tensor, weight: Tensor, loss_name: str, class_num: int = 2) -> Tensor:
        if loss_name in ['bcw', 'uw']:
            loss = self.wm_criterion.forward(output, label, weight)
        return loss


class WeightMapLoss(nn.Module):
    """
        Calculate weighted loss with weight maps
    """

    def forward(self, pred: Tensor, target: Tensor, weight_maps: Tensor, eps: float = 1e-20) -> Tensor:
        class_num = weight_maps.size()[1]
        mask = target.float()
        logit = torch.softmax(pred, dim=1)
        loss = 0
        weight_maps = weight_maps.float()
        for idx in range(class_num):
            if weight_maps.dim() == 4:
                loss += -1 * weight_maps[:, idx, :, :] * (torch.log(logit[:, idx, :, :]) + eps)

            elif weight_maps.dim() == 5:
                loss += -1 * weight_maps[:, idx, :, :, :] * (torch.log(logit[:, idx, :, :, :]) + eps)
        return loss.sum() / weight_maps.sum()
