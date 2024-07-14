import torch
from torch.nn import init
from torch.nn.parallel import DistributedDataParallel


'''
Paper: `CycleGAN and pix2pix`
'''
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


'''
Paper: `MASA-SR: Matching Acceleration and Spatial Adaptation for Reference-Based Image Super-Resolution`
'''
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(net):
        for name, m in net.named_modules():
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

def init_net(net, gpu_ids=[], device=None, dist=False, init_type='normal', init_gain=0.02):
    if len(gpu_ids) > 0:
        if not torch.cuda.is_available():
            raise AssertionError
        net.to(device)
        if dist:
            net = DistributedDataParallel(net, device_ids=[torch.cuda.current_device()])
        else:
            net = torch.nn.DataParallel(net, gpu_ids)
    # init_weights(net, init_type, gain=init_gain)
    return net


'''
Paper: `Exploring Spatial-Temporal Multi-Frequency Analysis for High-Fidelity and Temporal-Consistency Video Prediction`
'''
def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    elif init_type == 'STMF':
        net.apply(weights_init_stmf)
    elif init_type == 'zeros':
        net.apply(weights_init_zeros)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_normal(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_zeros(m):
    classname = m.__class__.__name__
    # pdb.set_trace()
    if classname.find('Conv') != -1 and classname != 'ConvLstmCell':
        # init.xavier_normal(m.weight.data, gain=1)
        # init.uniform(m.weight.data, 0.0, 0.02)
        init.uniform_(m.weight.data, 0.0, 0.0001)
        init.constant_(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
        # m.bias.data = m.bias.data.double()
        # pdb.set_trace()
    elif classname.find('Linear') != -1:
        init.uniform_(m.weight.data, 0.0, 0.0001)
        init.constant_(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
        # m.bias.data = m.bias.data.double()


def weights_init_stmf(m):
    classname = m.__class__.__name__

    # pdb.set_trace()
    if classname.find('Conv') != -1 and classname != 'ConvLstmCell':
        init.xavier_normal_(m.weight.data, gain=1)
        # init.constant(m.weight.data, 0.0)
        init.constant_(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
        # m.bias.data = m.bias.data.double()
        # pdb.set_trace()
    elif classname.find('Linear') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        # m.weight.data = m.weight.data.double()
        # m.bias.data = m.bias.data.double()