import torch
import torch.nn as nn
import torch.nn.functional as F


'''
cross entropy
thresholds = [0.25, 0.375, 0.5, 0.625]
weights = torch.FloatTensor([2, 3, 6, 10, 30])
midlevalue = [0.1875, 0.3125, 0.4375, 0.5625, 0.8125]
Initialize: WeightedCrossEntropyLoss(thresholds, weights, 5),  prob = ProbToPixel(midlevalue)
Paper: `Spatiotemporal Convolutional LSTM for Radar Echo Extrapolation`
'''
class WeightedCrossEntropyLoss(nn.Module):
    # weight should be a 1D Tensor assigning weight to each of the classes.
    def __init__(self, thresholds, weight=None, LAMBDA=None):
        super().__init__()
        # 每个类别的权重，使用原文章的权重。
        self._weight = weight
        # 每一帧 Loss 递进参数
        self._lambda = LAMBDA
        # thresholds: 雷达反射率
        self._thresholds = thresholds

    # input: output prob, b*s*C*H*W
    # target: b*s*1*H*W, original data, range [0, 1]
    # mask: S*B*1*H*W
    def forward(self, inputs, targets):
        # assert input.size(0) == cfg.HKO.BENCHMARK.OUT_LEN

        # F.cross_entropy should be B*C*S*H*W
        inputs = inputs.permute((0, 2, 1, 3, 4))

        # B*S*H*W
        targets = targets.squeeze(2)
        class_index = torch.zeros_like(targets).long()

        thresholds = self._thresholds
        class_index[...] = 0

        for i, threshold in enumerate(thresholds):
            i = i + 1
            class_index[targets >= threshold] = i

        if self._weight is not None:
            self._weight = self._weight.to(targets.device)
            error = F.cross_entropy(inputs, class_index, self._weight, reduction='none')
        else:
            error = F.cross_entropy(inputs, class_index, reduction='none')

        if self._lambda is not None:
            B, S, H, W = error.size()
            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)

            if torch.cuda.is_available():
                w = w.to(targets.device)

            # B, H, W, S
            error = (w * error.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # S*B*1*H*W
        error = error.permute(0, 1, 2, 3).unsqueeze(2)

        return torch.mean(error.float())