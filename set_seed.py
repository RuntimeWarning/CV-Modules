import torch
import random
import numpy as np


'''
Paper: `VPTR: Efficient Transformers for Video Prediction`
'''
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


'''
Paper: `A Dynamic Multi-Scale Voxel Flow Network for Video Prediction`
'''
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True