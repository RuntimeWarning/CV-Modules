from torch import nn


'''
Paper: `MoDeRNN: Towards Fine-grained Motion Details for Spatiotemporal Predictive Learning`
'''
class MSEL1Loss(nn.Module):
    def __init__(self, size_average=False, reduce=True, alpha=1.0):
        super(MSEL1Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.alpha = alpha

        self.mse_criterion = nn.MSELoss(size_average=self.size_average, reduce=self.reduce)
        self.l1_criterion = nn.L1Loss(size_average=self.size_average, reduce=self.reduce)

    def __call__(self, input, target):
        mse_loss = self.mse_criterion(input, target)
        l1_loss = self.l1_criterion(input, target)
        loss = mse_loss + self.alpha * l1_loss
        return loss / 2.