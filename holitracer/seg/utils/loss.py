import torch
import torch.nn as nn


class SegmentationLosses(object):
    def __init__(self, weight=None, reduction='mean', ignore_index=-100, device=torch.device("cuda")):
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduction = reduction
        self.device = device

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.reduction).to(self.device)

        loss = criterion(logit, target.long())
        losses = {"loss_tot": loss}
        return losses

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        reduction=self.reduction).to(self.device)
        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt
        losses = {"loss_tot": loss}
        return losses
