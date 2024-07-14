import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

class ConvGRUCell(nn.Module):
    def __init__(self, in_channel, num_hidden, filter_size, stride, sn_eps=0.0001):
        super(ConvGRUCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._sn_eps = sn_eps

        self.read_gate_conv = spectral_norm(
            torch.nn.Conv2d(
                in_channels=in_channel * 2,
                out_channels=num_hidden,
                kernel_size=filter_size,
                stride=stride,
                padding=self.padding,
            ),
            eps=sn_eps,
        )
        self.update_gate_conv = spectral_norm(
            torch.nn.Conv2d(
                in_channels=in_channel * 2,
                out_channels=num_hidden,
                kernel_size=filter_size,
                stride=stride,
                padding=self.padding,
            ),
            eps=sn_eps,
        )
        self.output_conv = spectral_norm(
            torch.nn.Conv2d(
                in_channels=in_channel * 2,
                out_channels=num_hidden,
                kernel_size=filter_size,
                stride=stride,
                padding=self.padding,
            ),
            eps=sn_eps,
        )

    def forward(self, x, prev_state):

        # Concatenate the inputs and previous state along the channel axis.
        xh = torch.cat([x, prev_state], dim=1)

        # Read gate of the GRU.
        read_gate = torch.sigmoid(self.read_gate_conv(xh))

        # Update gate of the GRU.
        update_gate = torch.sigmoid(self.update_gate_conv(xh))

        # Gate the inputs.
        gated_input = torch.cat([x, read_gate * prev_state], dim=1)

        # Gate the cell and state / outputs.
        c = torch.relu(self.output_conv(gated_input))
        out = update_gate * prev_state + (1.0 - update_gate) * c
        new_state = out

        return out, new_state


'''
Paper: `Deep Animation Video Interpolation in the Wild`
'''
class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h