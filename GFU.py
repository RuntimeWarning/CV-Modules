import torch
import torch.nn as nn


'''
Gate Fusion Unit (GFU)
Z, R = split(σ(Conv(cat(l, g))))
out = Conv(g) ∗ Z + Conv(l) ∗ R
Paper: `Rainformer: Features Extraction Balanced Network for Radar-Based Precipitation Nowcasting`
'''
class GFU(nn.Module):
    def __init__(self, channel, h_w, kernel_size=3, stride=1, padding=1):
        super(GFU, self).__init__()
        height, width = h_w
        self.conv_1 = nn.Sequential(
            nn.Conv2d(channel * 2, channel * 2, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LayerNorm([channel * 2, height, width]),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LayerNorm([channel, height, width]),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LayerNorm([channel, height, width]),
        )

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, g, l):
        cat_1 = self.conv_1(torch.cat((g, l), dim=1))
        Z, R = torch.chunk(cat_1, 2, dim=1)
        Z = torch.sigmoid(Z)
        R = torch.sigmoid(R)

        H = Z * g + R * l

        return H