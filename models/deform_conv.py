# deform_conv.py

import torch
import torch.nn as nn
try:
    from torchvision.ops import DeformConv2d as TVDeformConv2d
except ImportError:
    raise ImportError("torchvision.ops.DeformConv2d not found. Please update torchvision to a version that supports DeformConv2d.")

class DeformConv2d(nn.Module):
    """
    Minimal wrapper around torchvision's DeformConv2d.
    This layer applies deformable convolution to the input feature map.

    Arguments:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Default: False.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(DeformConv2d, self).__init__()
        self.deform_conv = TVDeformConv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )

    def forward(self, x, offset):
        # x: input feature map, offset: offsets for deformable convolution
        return self.deform_conv(x, offset)
