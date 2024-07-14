import torch
import numpy as np
import torch.nn as nn
from math import sqrt


'''
Paper: `PredRANN: The Spatiotemporal Attention Convolution Recurrent Neural Network for Precipitation Nowcasting`
'''
class TimeDistribution(nn.Module):
    def __init__(self,model):
        super(TimeDistribution, self).__init__()
        self.model = model

    def forward(self, input):
        t_length = input.shape[1]
        outputs = []
        for t in range(t_length):
            outputs.append(self.model(input[:,t]))
        outputs = torch.stack(outputs,1)
        return outputs


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        # the B is the batch size;
        # the L is the length of sequence;
        # the H is the number of head;
        # the E is the input dimension;

        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class TransformerCell(nn.Module):
    def __init__(self,qin_channels,kvin_channels, heads, head_channels, width):
        super(TransformerCell, self).__init__()
        self.qin_channels = qin_channels
        self.kvin_channels = kvin_channels
        self.heads = heads
        self.head_channels = head_channels
        self.width = width
        self.inner_channels = self.head_channels * self.heads
        self.channel_attn = FullAttention(mask_flag=False, factor=5, attention_dropout=0.2)
        self.k_projection = TimeDistribution(
            model=nn.Conv2d(
                in_channels = self.kvin_channels,
                out_channels = self.inner_channels,
                kernel_size=1,
                padding=0
            )
        )
        self.v_projection = TimeDistribution(
            model=nn.Conv2d(
                in_channels=self.kvin_channels,
                out_channels=self.inner_channels,
                kernel_size=1,
                padding=0
            )
        )
        self.q_projection = nn.Conv2d(
                in_channels = self.qin_channels,
                out_channels = self.inner_channels,
                kernel_size=1,
                padding=0
            )
        self.output_projection = nn.Conv2d(
                in_channels = self.inner_channels,
                out_channels = self.qin_channels,
                kernel_size = 3,
                padding = 1
            )

        self.norm = nn.LayerNorm([qin_channels, width, width])


    def forward(self, in_query, key, value):
        if type(key)==type([]):
            key = torch.stack(key,1)
            value = torch.stack(value, 1)
        query = self.q_projection(in_query)
        key = self.k_projection(key)
        value = self.v_projection(value)

        B,T,_,H,W = key.shape

        query = query.view(B, 1, self.heads, self.head_channels, H*W).permute((0,1,3,2,4))
        key = key.view(B, T, self.heads, self.head_channels, H*W).permute((0,1,3,2,4))
        value = value.view(B, T, self.heads, self.head_channels, H*W).permute((0,1,3,2,4))

        query = query.reshape(B, 1*self.head_channels,self.heads, H*W)
        key = key.reshape(B, T*self.head_channels, self.heads, H*W)
        value = value.reshape(B, T*self.head_channels, self.heads,  H*W)

        s_attn = self.channel_attn(query, key, value,None)
        s_attn = s_attn.view(B, 1, self.head_channels,self.heads, H, W)
        s_attn = s_attn.reshape(B, 1*self.heads*self.head_channels, H, W)

        output = self.output_projection(s_attn)
        output = self.norm(in_query + output)

        return output