"""
Skilful precipitation nowcasting using deep generative models of radar, from DeepMind
https://arxiv.org/abs/2104.00954
"""

import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    """
    Spatial attention module: for latent conditioning stack
    """
    def __init__(self, in_channels=192, out_channels=192, ratio_kq=8, ratio_v=8, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.ratio_kq = ratio_kq
        self.ratio_v = ratio_v
        self.conv_q = nn.Conv2d(in_channels, out_channels//ratio_kq, 1, 1, 0, bias=False)
        self.conv_k = nn.Conv2d(in_channels, out_channels//ratio_kq, 1, 1, 0, bias=False)
        self.conv_v = nn.Conv2d(in_channels, out_channels//ratio_v, 1, 1, 0, bias=False)
        self.conv_out = nn.Conv2d(out_channels//ratio_v, out_channels, 1, 1, 0, bias=False)
    
    def einsum(self, q, k, v):
        # org shape = B, C, H, W
        k = k.view(k.shape[0], k.shape[1], -1) # B, C, H*W
        v = v.view(v.shape[0], v.shape[1], -1) # B, C, H*W
        beta = torch.einsum("bchw, bcL->bLhw", q, k)
        beta = torch.softmax(beta, dim=1)
        out = torch.einsum("bLhw, bcL->bchw", beta, v)
        return out

    def forward(self, x):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        
        # the question is whether x should be preserved or just attn
        out = self.einsum(q, k, v)
        out = self.conv_out(out)
        return x + out
    

# OpenSTL: Open-source Toolbox for SpatioTemporal Predictive Learning
class AttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size, dilation=3):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, 2*dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)           # depth-wise conv
        attn = self.conv_spatial(attn) # depth-wise dilation convolution
        
        f_g = self.conv1(attn)
        split_dim = f_g.shape[1] // 2
        f_x, g_x = torch.split(f_g, split_dim, dim=1)
        return torch.sigmoid(g_x) * f_x


class SpatialAttention(nn.Module):
    """A Spatial Attention block for SimVP"""

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.activation = nn.GELU()                          # GELU
        self.spatial_gating_unit = AttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x