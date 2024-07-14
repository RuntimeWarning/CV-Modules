import torch
import torch.nn as nn


'''
Paper: `VPTR: Efficient Transformers for Video Prediction`
'''
class GDL(nn.Module):
    def __init__(self, alpha = 1, temporal_weight = None):
        """
        Args:
            alpha: hyper parameter of GDL loss, float
            temporal_weight: penalty for loss at different time step, Tensor with length T
        """
        super().__init__()
        self.alpha = alpha
        self.temporal_weight = temporal_weight

    def __call__(self, gt, pred):
        """
        pred --- tensor with shape (B, T, ...)
        gt --- tensor with shape (B, T, ...)
        """
        gt_shape = gt.shape
        if len(gt_shape) == 5:
            B, T, _, _, _ = gt.shape
        elif len(gt_shape) == 6: #for warped frames, (N, num_future_frames, num_past_frames, C, H, W)
            B, T, TP, _, _, _ = gt.shape
        gt = gt.flatten(0, -4)
        pred = pred.flatten(0, -4)

        gt_i1 = gt[:, :, 1:, :]
        gt_i2 = gt[:, :, :-1, :]
        gt_j1 = gt[:, :, :, :-1]
        gt_j2 = gt[:, :, :, 1:]

        pred_i1 = pred[:, :, 1:, :]
        pred_i2 = pred[:, :, :-1, :]
        pred_j1 = pred[:, :, :, :-1]
        pred_j2 = pred[:, :, :, 1:]

        term1 = torch.abs(gt_i1 - gt_i2)
        term2 = torch.abs(pred_i1 - pred_i2)
        term3 = torch.abs(gt_j1 - gt_j2)
        term4 = torch.abs(pred_j1 - pred_j2)

        if self.alpha != 1:
            gdl1 = torch.pow(torch.abs(term1 - term2), self.alpha)
            gdl2 = torch.pow(torch.abs(term3 - term4), self.alpha)
        else:
            gdl1 = torch.abs(term1 - term2)
            gdl2 = torch.abs(term3 - term4)
        
        if self.temporal_weight is not None:
            assert self.temporal_weight.shape[0] == T, "Mismatch between temporal_weight and predicted sequence length"
            w = self.temporal_weight.to(gdl1.device)
            _, C, H, W = gdl1.shape
            _, C2, H2, W2= gdl2.shape
            if len(gt_shape) == 5:
                gdl1 = gdl1.reshape(B, T, C, H, W)
                gdl2 = gdl2.reshape(B, T, C2, H2, W2)
                gdl1 = gdl1 * w[None, :, None, None, None]
                gdl2 = gdl2 * w[None, :, None, None, None]
            elif len(gt_shape) == 6:
                gdl1 = gdl1.reshape(B, T, TP, C, H, W)
                gdl2 = gdl2.reshape(B, T, TP, C2, H2, W2)
                gdl1 = gdl1 * w[None, :, None, None, None, None]
                gdl2 = gdl2 * w[None, :, None, None, None, None]

        #gdl1 = gdl1.sum(-1).sum(-1)
        #gdl2 = gdl2.sum(-1).sum(-1)

        #gdl_loss = torch.mean(gdl1 + gdl2)
        gdl1 = gdl1.mean()
        gdl2 = gdl2.mean()
        gdl_loss = gdl1 + gdl2
        
        return gdl_loss



class GradientDifferenceLoss(nn.Module):

    def __init__(self, alpha: int = 2):
        super(GradientDifferenceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        t1 = torch.pow(
            torch.abs(
                torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
                - torch.abs(y[:, :, :, 1:, :] - y[:, :, :, :-1, :])
            ),
            self.alpha,
        )
        t2 = torch.pow(
            torch.abs(
                torch.abs(x[:, :, :, :, :-1] - x[:, :, :, :, 1:])
                - torch.abs(y[:, :, :, :, :-1] - y[:, :, :, :, 1:])
            ),
            self.alpha,
        )
        loss = t1 + t2
        return loss
    

'''
Paper: `Reduce Information Loss in Transformers for Pluralistic Image Inpainting`
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageGradientLoss(nn.Module):
    def __init__(self):
        super(ImageGradientLoss, self).__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, fake, real):
        fake_dx, fake_dy = self.gradient(fake)
        real_dx, real_dy = self.gradient(real.detach())
        g_loss = (self.loss(fake_dx,real_dx)+self.loss(fake_dy,real_dy))/2
        return g_loss
    
    def gradient(self, x):
        # gradient step=1
        l = x
        r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        t = x
        b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        dx, dy = torch.abs(r - l), torch.abs(b - t)
        # dx will always have zeros in the last column, r-l
        # dy will always have zeros in the last row,    b-t
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy