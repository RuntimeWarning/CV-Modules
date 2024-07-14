import torch.nn as nn
import torch.nn.functional as F


'''
Paper: `MASA-SR: Matching Acceleration and Spatial Adaptation for Reference-Based Image Super-Resolution`
'''
class ResidualBlock(nn.Module):
    def __init__(self, nf, kernel_size=3, stride=1, padding=1, dilation=1, act='relu'):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))

        return out + x


'''
Paper: `Deep Animation Video Interpolation in the Wild`
'''
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)



class ResBlock(nn.Module):
    """
    Basic residual block for SRNTT.

    Parameters
    ---
    n_filters : int, optional
        a number of filters.

    self.n_res_block = nn.Sequential(
        *[ResBlock(ngf) for _ in range(n_blocks)],
    )
    """

    def __init__(self, n_filters=64):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
        )

    def forward(self, x):
        return self.body(x) + x
    

'''
Paper: `Reduce Information Loss in Transformers for Pluralistic Image Inpainting`
'''
class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False, with_instance_norm=True):
        super(ResnetBlock, self).__init__()
        conv_block_ = [
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        ]
        conv_block = []
        for m in conv_block_:
            if isinstance(m, nn.InstanceNorm2d):
                if with_instance_norm:
                    conv_block.append(m)
            else:
                conv_block.append(m)
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


'''
Paper: `MOSO: Decomposing MOtion, Scene and Object for Video Prediction`
'''
class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, add_BN=False):
        super(ResidualStack, self).__init__()

        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._layers = nn.ModuleList()
        if add_BN is True:
            for i in range(num_residual_layers):
                curlayer = nn.Sequential(
                    nn.Conv2d(num_hiddens, num_residual_hiddens,
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(num_residual_hiddens),
                    nn.GELU(),
                    nn.Conv2d(num_residual_hiddens, num_hiddens,
                              kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(num_hiddens),
                    nn.GELU(),
                )
                self._layers.append(curlayer)
        else:
            for i in range(num_residual_layers):
                curlayer = nn.Sequential(
                    nn.Conv2d(num_hiddens, num_residual_hiddens,
                              kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.Conv2d(num_residual_hiddens, num_hiddens,
                              kernel_size=1, stride=1, padding=0),
                    nn.GELU()
                )
                self._layers.append(curlayer)

    def forward(self, inputs):
        h = inputs
        for layer in self._layers:
            z = layer(h)
            h = h + z
        return F.gelu(h)