import torch
from globalAttention import globalAttention

from fvcore.nn.flop_count import flop_count




def params_flops(net, x, **kwargs):
    net_params = sum(map(lambda x: x.numel(), net.parameters()))/10**6
    flop_dict, _ = flop_count(net, x, **kwargs)
    sumflops = 0
    for i in flop_dict: 
        sumflops = sumflops + flop_dict[i]
    return net_params, sumflops

if __name__ == '__main__':

    x = torch.randn(1, 5, 64, 64, 64)
    attention = globalAttention()
    attn_params, attn_flops =  params_flops(attention, x)
    print("globalAttention", "Params [M]:", attn_params, "Flops [G]:", attn_flops)
