import torch
from torch import nn
from torch.nn import functional as F


'''
Paper: `VSR-Transformer`
'''
class globalAttention(nn.Module):
    def __init__(self, num_feat=64, patch_size=8, heads=1):
        super(globalAttention, self).__init__()
        self.heads = heads
        self.dim = patch_size ** 2 * num_feat
        self.hidden_dim = self.dim // heads
        self.num_patch = (64 // patch_size) ** 2
        
        self.to_q = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, groups=num_feat) 
        self.to_k = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, groups=num_feat)
        self.to_v = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1)

        self.conv = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1)

        self.feat2patch = torch.nn.Unfold(kernel_size=patch_size, padding=0, stride=patch_size)
        self.patch2feat = torch.nn.Fold(output_size=(64, 64), kernel_size=patch_size, padding=0, stride=patch_size)

    def forward(self, x):
        b, t, c, h, w = x.shape                                # B, 5, 64, 64, 64
        H, D = self.heads, self.dim
        n, d = self.num_patch, self.hidden_dim

        q = self.to_q(x.view(-1, c, h, w))                     # [B*5, 64, 64, 64]    
        k = self.to_k(x.view(-1, c, h, w))                     # [B*5, 64, 64, 64]   
        v = self.to_v(x.view(-1, c, h, w))                     # [B*5, 64, 64, 64]

        unfold_q = self.feat2patch(q)                          # [B*5, 8*8*64, 8*8]
        unfold_k = self.feat2patch(k)                          # [B*5, 8*8*64, 8*8]  
        unfold_v = self.feat2patch(v)                          # [B*5, 8*8*64, 8*8] 

        unfold_q = unfold_q.view(b, t, H, d, n)                # [B, 5, H, 8*8*64/H, 8*8]
        unfold_k = unfold_k.view(b, t, H, d, n)                # [B, 5, H, 8*8*64/H, 8*8]
        unfold_v = unfold_v.view(b, t, H, d, n)                # [B, 5, H, 8*8*64/H, 8*8]

        unfold_q = unfold_q.permute(0,2,3,1,4).contiguous()    # [B, H, 8*8*64/H, 5, 8*8]
        unfold_k = unfold_k.permute(0,2,3,1,4).contiguous()    # [B, H, 8*8*64/H, 5, 8*8]
        unfold_v = unfold_v.permute(0,2,3,1,4).contiguous()    # [B, H, 8*8*64/H, 5, 8*8]

        unfold_q = unfold_q.view(b, H, d, t*n)                 # [B, H, 8*8*64/H, 5*8*8]
        unfold_k = unfold_k.view(b, H, d, t*n)                 # [B, H, 8*8*64/H, 5*8*8]
        unfold_v = unfold_v.view(b, H, d, t*n)                 # [B, H, 8*8*64/H, 5*8*8]

        attn = torch.matmul(unfold_q.transpose(2,3), unfold_k) # [B, H, 5*8*8, 5*8*8]
        attn = attn * (d ** (-0.5))                            # [B, H, 5*8*8, 5*8*8]
        attn = F.softmax(attn, dim=-1)                         # [B, H, 5*8*8, 5*8*8]

        attn_x = torch.matmul(attn, unfold_v.transpose(2,3))   # [B, H, 5*8*8, 8*8*64/H]
        attn_x = attn_x.view(b, H, t, n, d)                    # [B, H, 5, 8*8, 8*8*64/H]
        attn_x = attn_x.permute(0, 2, 1, 4, 3).contiguous()    # [B, 5, H, 8*8*64/H, 8*8]
        attn_x = attn_x.view(b*t, D, n)                        # [B*5, 8*8*64, 8*8]
        feat = self.patch2feat(attn_x)                         # [B*5, 64, 64, 64]
        
        out = self.conv(feat).view(x.shape)                    # [B, 5, 64, 64, 64]
        out += x                                               # [B, 5, 64, 64, 64]

        return out