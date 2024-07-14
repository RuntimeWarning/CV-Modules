import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


'''
Paper: `VPTR: Efficient Transformers for Video Prediction`
'''
def temporal_weight_func(T):
    t = torch.linspace(0, T-1, T)
    beta = np.log(T)/(T-1)
    w = torch.exp(beta * t)

    return w

class L1Loss(nn.Module):
    def __init__(self, temporal_weight = None, norm_dim = None):
        """
        Args:
            temporal_weight: penalty for loss at different time step, Tensor with length T
        """
        super().__init__()
        self.temporal_weight = temporal_weight
        self.norm_dim = norm_dim
    
    def __call__(self, gt, pred):
        """
        pred --- tensor with shape (B, T, ...)
        gt --- tensor with shape (B, T, ...)
        """
        if self.norm_dim is not None:
            gt = F.normalize(gt, p = 2, dim = self.norm_dim)
            pred = F.normalize(pred, p = 2, dim = self.norm_dim)

        se = torch.abs(pred - gt)
        if self.temporal_weight is not None:
            w = self.temporal_weight.to(se.device)
            if len(se.shape) == 5:
                se = se * w[None, :, None, None, None]
            elif len(se.shape) == 6:
                se = se * w[None, :, None, None, None, None] #for warped frames, (N, num_future_frames, num_past_frames, C, H, W)
        mse = se.mean()
        return mse

class MSELoss(nn.Module):
    def __init__(self, temporal_weight = None, norm_dim = None):
        """
        Args:
            temporal_weight: penalty for loss at different time step, Tensor with length T
        """
        super().__init__()
        self.temporal_weight = temporal_weight
        self.norm_dim = norm_dim
    
    def __call__(self, gt, pred):
        """
        pred --- tensor with shape (B, T, ...)
        gt --- tensor with shape (B, T, ...)
        """
        if self.norm_dim is not None:
            gt = F.normalize(gt, p = 2, dim = self.norm_dim)
            pred = F.normalize(pred, p = 2, dim = self.norm_dim)

        se = torch.square(pred - gt)
        if self.temporal_weight is not None:
            w = self.temporal_weight.to(se.device)
            if len(se.shape) == 5:
                se = se * w[None, :, None, None, None]
            elif len(se.shape) == 6:
                se = se * w[None, :, None, None, None, None] #for warped frames, (N, num_future_frames, num_past_frames, C, H, W)
        mse = se.mean()
        return mse


'''
Paper: `Deep learning for precipitation nowcasting: a benchmark and a new model`
'''
class Weighted_mse_mae(nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=1.0, NORMAL_LOSS_GLOBAL_SCALE=0.00005, LAMBDA=None):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self._lambda = LAMBDA

    def forward(self, input, target, mask):
        balancing_weights = (1, 1, 2, 5, 10, 30)
        weights = torch.ones_like(input) * balancing_weights[0]
        thresholds = np.array([0.5, 2, 5, 10, 30])
        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (target >= threshold).float()
        weights = weights * mask.float()
        # input: S*B*1*H*W
        # error: S*B
        mse = torch.sum(weights * ((input-target)**2), (2, 3, 4))
        mae = torch.sum(weights * (torch.abs((input-target))), (2, 3, 4))
        if self._lambda is not None:
            S, B = mse.size()
            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)
            if torch.cuda.is_available():
                w = w.to(mse.get_device())
            mse = (w * mse.permute(1, 0)).permute(1, 0)
            mae = (w * mae.permute(1, 0)).permute(1, 0)
        return self.NORMAL_LOSS_GLOBAL_SCALE * (self.mse_weight*torch.mean(mse) + self.mae_weight*torch.mean(mae))


'''
Paper: `Rainformer: Features Extraction Balanced Network for Radar-Based Precipitation Nowcasting`
2, 5, 10, 30 is the thresholds of the precipitation intensity
'''
class BMAEloss(nn.Module):
    def __init__(self):
        super(BMAEloss, self).__init__()

    def fundFlag(self, a, n, m):
        flag_1 = (a >= n).int()
        flag_2 = (a < m).int()
        flag_3 = flag_1 + flag_2
        return flag_3 == 2

    def forward(self, pred, y):
        mask = torch.zeros(y.shape).cuda()
        mask[y < 2] = 1
        mask[self.fundFlag(y, 2, 5)] = 2
        mask[self.fundFlag(y, 5, 10)] = 5
        mask[self.fundFlag(y, 10, 30)] = 10
        mask[y > 30] = 30
        return torch.sum(mask * torch.abs(y - pred))


def fundFlag(a, n, m):
    flag_1 = np.uint8(a >= n)
    flag_2 = np.uint8(a < m)
    flag_3 = flag_1 + flag_2
    return flag_3 == 2

def B_mse(a, b):
    mask = np.zeros(a.shape)
    mask[a < 2] = 1
    mask[fundFlag(a, 2, 5)] = 2
    mask[fundFlag(a, 5, 10)] = 5
    mask[fundFlag(a, 10, 30)] = 10
    mask[a > 30] = 30
    n = a.shape[0] * b.shape[0]
    mse = np.sum(mask * ((a - b) ** 2)) / n
    return mse

def B_mae(a, b):
    mask = np.zeros(a.shape)
    mask[a < 2] = 1
    mask[fundFlag(a, 2, 5)] = 2
    mask[fundFlag(a, 5, 10)] = 5
    mask[fundFlag(a, 10, 30)] = 10
    mask[a > 30] = 30
    n = a.shape[0] * b.shape[0]
    mae = np.sum(mask * np.abs(a - b)) / n
    return mae