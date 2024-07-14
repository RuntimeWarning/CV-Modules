import torch
from torch import Tensor


'''
Paper: `VPTR: Efficient Transformers for Video Prediction`
'''
def MSEScore(x: Tensor, y: Tensor) -> Tensor:
    """
    Comput the average PSNR between two batch of images.
    x: input image, Tensor with shape (N, C, H, W)
    y: input image, Tensor with shape (N, C, H, W)
    data_range: the maximum pixel value range of input images, used to normalize
                pixel values to [0,1], default is 1.0
    """
    mse = torch.sum((x-y)**2, dim = (1, 2, 3))

    return torch.mean(mse).item()