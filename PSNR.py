import torch
from torch import Tensor
from typing import Union


'''
Paper: `VPTR: Efficient Transformers for Video Prediction`
'''
def PSNR(x: Tensor, y: Tensor, data_range: Union[float, int] = 1.0) -> Tensor:
    """
    Comput the average PSNR between two batch of images.
    x: input image, Tensor with shape (N, C, H, W)
    y: input image, Tensor with shape (N, C, H, W)
    data_range: the maximum pixel value range of input images, used to normalize
                pixel values to [0,1], default is 1.0
    """

    EPS = 1e-8
    x = x/float(data_range)
    y = y/float(data_range)

    mse = torch.mean((x-y)**2, dim = (1, 2, 3))
    score = -10*torch.log10(mse + EPS)

    return torch.mean(score).item()