import torch
import torch.nn as nn


'''
Paper: `A Dynamic Multi-Scale Voxel Flow Network for Video Prediction`
'''
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6, mode=None):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.mode = mode

    def forward(self, x, y, mask=None):
        N = x.size(1)
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps)
        if mask is not None:
            loss = loss * mask
        if self.mode == 'sum':
            loss = torch.sum(loss) / N
        else:
            loss = loss.mean()
        return loss