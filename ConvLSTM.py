import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride):
        super(LSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.d = num_hidden * height * width
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, height, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, height, width])
        )

    def forward(self, x_t, h_t, c_t):

        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t
        o_t = torch.sigmoid(o_x + o_h + c_new)
        h_new = o_t * torch.tanh(c_new)

        return c_new, h_new


'''
Paper: `Video Prediction Recalling Long-term Motion Context via Memory Alignment Learning`
'''
class NPUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(NPUnit, self).__init__()
        same_padding = int((kernel_size[0]-1)/2)
        self.conv2d_x = nn.Conv2d(in_channels=in_channels, out_channels=4*out_channels,
                                  kernel_size=kernel_size, stride=1, padding=same_padding, bias=True)
        self.conv2d_h = nn.Conv2d(in_channels=out_channels, out_channels=4*out_channels,
                                  kernel_size=kernel_size, stride=1, padding=same_padding, bias=True)

    def forward(self, x, h, c):
        x_after_conv = self.conv2d_x(x)
        h_after_conv = self.conv2d_h(h)
        xi, xc, xf, xo = torch.chunk(x_after_conv, 4, dim=1)
        hi, hc, hf, ho = torch.chunk(h_after_conv, 4, dim=1)

        it = torch.sigmoid(xi+hi)
        ft = torch.sigmoid(xf+hf)
        new_c = (ft*c)+(it*torch.tanh(xc+hc))
        ot = torch.sigmoid(xo+ho)
        new_h = ot*torch.tanh(new_c)

        return new_h, new_c