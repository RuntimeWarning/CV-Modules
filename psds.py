import torch
import numpy as np
from imageio.v2 import imread
import scipy.stats as stats


'''
Paper: `Towards a More Realistic and Detailed Deep-Learning-Based Radar Echo Extrapolation Method`
'''
def fft_score(images1, images2, thresh=0):

    N, C, H, W = images1.shape

    kx = torch.fft.fftfreq(W) * W
    ky = torch.fft.fftfreq(H) * H
    kyx = torch.meshgrid(ky, kx)

    knorm = torch.sqrt(kyx[0]**2 + kyx[1]**2)
    knorm = knorm[None, None].repeat(N, C, 1, 1)

    images1 = torch.fft.fftn(images1, dim=(2,3)).abs() ** 2
    images2 = torch.fft.fftn(images2, dim=(2,3)).abs() ** 2

    images1 = images1.flatten().cpu().numpy()
    images2 = images2.flatten().cpu().numpy()
    knorm = knorm.flatten().cpu().numpy()

    kbins = np.arange(0.5, min(H, W) // 2 + 1, 1.)

    v1, _, _ = stats.binned_statistic(knorm, images1, statistic = "sum", bins = kbins)
    v2, _, _ = stats.binned_statistic(knorm, images2, statistic = "sum", bins = kbins)

    bins = np.arange(thresh, min(H, W) // 2, 1)

    if np.sum(v2[bins]) == 0:
        return 0.0

    score = np.abs(1.0 - np.sum(v1[bins]) / np.sum(v2[bins]))
    return score

if __name__ == '__main__':
    images1 = torch.from_numpy(np.array(imread(r'C:\Users\15244\Desktop\ob_100001_1.png'))).unsqueeze(0).unsqueeze(0)
    images2 = torch.from_numpy(np.array(imread(r'C:\Users\15244\Desktop\ob_100001_2.png'))).unsqueeze(0).unsqueeze(0)
    score = fft_score(images1, images2)
    print(score)