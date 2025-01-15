from typing import Tuple, List, Union, Type

import numpy as np
import torch.nn
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list

class ModulatedConv2d(nn.Module):
    def __init__(self, conv_op: Type[_ConvNd], in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(ModulatedConv2d, self).__init__()
        self.conv = conv_op(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.gamma_fc = nn.Linear(32, out_channels)
        self.beta_fc = nn.Linear(32, out_channels)

    def forward(self, x, embedding):
        """
        x: input feature map [B, C_in, H, W]
        embedding: modulation vector [B, embedding_dim]
        """
        gamma = self.gamma_fc(embedding).unsqueeze(-1).unsqueeze(-1)  # [B, out_channels, 1, 1]
        beta = self.beta_fc(embedding).unsqueeze(-1).unsqueeze(-1)  # [B, out_channels, 1, 1]
        if len(x.shape) == 5: # 3D
            gamma = gamma.unsqueeze(-1) # [B, out_channels, 1, 1, 1]
            beta = beta.unsqueeze(-1) # [B, out_channels, 1, 1, 1]

        x = self.conv(x)  # [B, out_channels, H, W]
        x = gamma * x + beta  # Modulated feature map
        return x

class ModulatedConvDropoutNormReLU(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        super(ModulatedConvDropoutNormReLU, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []

        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)

        self.modulated_conv = ModulatedConv2d(
            conv_op,
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            bias=conv_bias,
        )

        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        if nonlin_first and (norm_op is not None and nonlin is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x, type_embeddings):
        temp = self.modulated_conv(x, type_embeddings)
        return self.all_modules(temp)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)

class ConvDropoutNormReLU(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        super(ConvDropoutNormReLU, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []

        self.conv = conv_op(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            bias=conv_bias,
        )
        ops.append(self.conv)

        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        if nonlin is not None:
            self.nonlin = nonlin(**nonlin_kwargs)
            ops.append(self.nonlin)

        if nonlin_first and (norm_op is not None and nonlin is not None):
            ops[-1], ops[-2] = ops[-2], ops[-1]

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x):
        return self.all_modules(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)


class StackedModulatedConvBlocks(nn.Module):
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        """

        :param conv_op:
        :param num_convs:
        :param input_channels:
        :param output_channels: can be int or a list/tuple of int. If list/tuple are provided, each entry is for
        one conv. The length of the list/tuple must then naturally be num_convs
        :param kernel_size:
        :param initial_stride:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        """
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs_without_embeddings = ConvDropoutNormReLU(
            conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias, norm_op,
            norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
        )
    
        if len(output_channels) > 1:
            self.convs_with_embeddings_1 = ModulatedConvDropoutNormReLU(
                conv_op, output_channels[1], output_channels[1], kernel_size, 1, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            )
        else:
            self.convs_with_embeddings_1 = None

        if len(output_channels) > 2:
            self.convs_with_embeddings_2 = ModulatedConvDropoutNormReLU(
                conv_op, output_channels[2], output_channels[2], kernel_size, 1, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            )
        else:
            self.convs_with_embeddings_2 = None

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x, type_embeddings):
        ret = x
        ret = self.convs_without_embeddings(ret)
        if self.convs_with_embeddings_1 is not None:
            ret = self.convs_with_embeddings_1(ret, type_embeddings)
        if self.convs_with_embeddings_2 is not None:
            ret = self.convs_with_embeddings_2(ret, type_embeddings)

        return ret

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        output = self.convs_without_embeddings.compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        if self.convs_with_embeddings_1 is not None:
            output += self.convs_with_embeddings_1.compute_conv_feature_map_size(size_after_stride)
        if self.convs_with_embeddings_2 is not None:
            output += self.convs_with_embeddings_2.compute_conv_feature_map_size(size_after_stride)
        return output


if __name__ == '__main__':
    data = torch.rand((1, 3, 40, 32))

    stx = StackedModulatedConvBlocks(3, nn.Conv2d, 24, 16, (3, 3), 2,
                            norm_op=nn.BatchNorm2d, nonlin=nn.ReLU, nonlin_kwargs={'inplace': True},
                            )
    model = nn.Sequential(ModulatedConvDropoutNormReLU(nn.Conv2d,
                                              3, 24, 3, 1, True, nn.BatchNorm2d, {}, None, None, nn.LeakyReLU,
                                              {'inplace': True}),
                          stx)
    import hiddenlayer as hl

    g = hl.build_graph(model, data,
                       transforms=None)
    g.save("network_architecture.pdf")
    del g

    stx.compute_conv_feature_map_size((40, 32))