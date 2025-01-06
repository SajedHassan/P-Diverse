    
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

"""
# --------------------------------------------
# Basic layers
# --------------------------------------------
"""
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2, padding_mode='zeros', dilation=1):
    """Define basic network layers, refer to Kai Zhang, https://github.com/cszn/KAIR"""
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode, dilation=dilation))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'P':
            L.append(nn.PReLU())
        else:
            raise NotImplementedError('Undefined type: {}'.format(t))
    return nn.Sequential(*L)

def maybe_convert_scalar_to_list(conv_op, scalar):
    """
    useful for converting, for example, kernel_size=3 to [3, 3, 3] in case of nn.Conv3d
    :param conv_op:
    :param scalar:
    :return:
    """
    if not isinstance(scalar, (tuple, list, np.ndarray)):
        if conv_op == nn.Conv2d:
            return [scalar] * 2
        elif conv_op == nn.Conv3d:
            return [scalar] * 3
        elif conv_op == nn.Conv1d:
            return [scalar] * 1
        else:
            raise RuntimeError("Invalid conv op: %s" % str(conv_op))
    else:
        return scalar

class Discriminator(nn.Module):
    def __init__(self, in_nc: int, nc: int, act_mode: str):
        super(Discriminator, self).__init__()

        conv0 = conv(in_nc, nc, kernel_size=7, padding=3, mode='C')
        conv1 = conv(nc, nc, kernel_size=4, stride=2, mode='C'+act_mode)
        # 48, 64
        conv2 = conv(nc, nc*2, kernel_size=3, stride=1, mode='C'+act_mode)
        conv3 = conv(nc*2, nc*2, kernel_size=4, stride=2, mode='C'+act_mode)
        # 24, 128
        conv4 = conv(nc*2, nc*4, kernel_size=3, stride=1, mode='C'+act_mode)
        conv5 = conv(nc*4, nc*4, kernel_size=4, stride=2, mode='C'+act_mode)
        # 12, 256
        conv6 = conv(nc*4, nc*8, kernel_size=3, stride=1, mode='C'+act_mode)
        conv7 = conv(nc*8, nc*8, kernel_size=4, stride=2, mode='C'+act_mode)
        # 6, 512
        conv8 = conv(nc*8, nc*8, kernel_size=3, stride=1, mode='C'+act_mode)
        conv9 = conv(nc*8, nc*8, kernel_size=4, stride=2, mode='C'+act_mode)
        # 3, 512
        self.features = nn.Sequential(*[conv0, conv1, conv2, conv3, conv4,
                                     conv5, conv6, conv7, conv8, conv9])

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 4)
        )

    def forward(self, x):
        x = F.interpolate(x, size=(96, 96), mode='bilinear', align_corners=False)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



if __name__ == '__main__':
    data = torch.rand((256, 2, 179, 164))

    model = Discriminator(2, 64, 'L')
    import hiddenlayer as hl

    g = hl.build_graph(model, data,
                       transforms=None)
    g.save("network_architecture.pdf")
    del g
