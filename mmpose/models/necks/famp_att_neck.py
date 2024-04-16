# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmpose.models.utils.ops import resize
from mmpose.registry import MODELS


@MODELS.register_module()
class FeatureMapAttProcessor(nn.Module):
    """A PyTorch module for selecting, concatenating, and rescaling feature
    maps.

    Args:
        select_index (Optional[Union[int, Tuple[int]]], optional): Index or
            indices of feature maps to select. Defaults to None, which means
            all feature maps are used.
        concat (bool, optional): Whether to concatenate the selected feature
            maps. Defaults to False.
        scale_factor (float, optional): The scaling factor to apply to the
            feature maps. Defaults to 1.0.
        apply_relu (bool, optional): Whether to apply ReLU on input feature
            maps. Defaults to False.
        align_corners (bool, optional): Whether to align corners when resizing
            the feature maps. Defaults to False.
    """

    def __init__(
        self,
        select_index: Optional[Union[int, Tuple[int]]] = None,
        concat: bool = False,
        scale_factor: float = 1.0,
        apply_relu: bool = False,
        align_corners: bool = False,
    ):
        super().__init__()

        if isinstance(select_index, int):
            select_index = (select_index, )
        self.select_index = select_index
        self.concat = concat

        assert (
            scale_factor > 0
        ), f'the argument `scale_factor` must be positive, ' \
           f'but got {scale_factor}'
        self.scale_factor = scale_factor
        self.apply_relu = apply_relu
        self.align_corners = align_corners

        # attention
        self.attention = attention()

    def forward(self, inputs: Union[Tensor, Sequence[Tensor]]
                ) -> Union[Tensor, List[Tensor]]:
        # print(inputs[-4].size())
        inputs = self.attention(inputs)
        if not isinstance(inputs, (tuple, list)):
            sequential_input = False
            inputs = [inputs]
        else:
            sequential_input = True

            if self.select_index is not None:
                inputs = [inputs[i] for i in self.select_index]

            if self.concat:
                inputs = self._concat(inputs)

        if self.apply_relu:
            inputs = [F.relu(x) for x in inputs]

        if self.scale_factor != 1.0:
            inputs = self._rescale(inputs)

        if not sequential_input:
            inputs = inputs[0]

        return inputs

    def _concat(self, inputs: Sequence[Tensor]) -> List[Tensor]:
        size = inputs[0].shape[-2:]
        resized_inputs = [
            resize(
                x,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners) for x in inputs
        ]
        return [torch.cat(resized_inputs, dim=1)]

    def _rescale(self, inputs: Sequence[Tensor]) -> List[Tensor]:
        rescaled_inputs = [
            resize(
                x,
                scale_factor=self.scale_factor,
                mode='bilinear',
                align_corners=self.align_corners,
            ) for x in inputs
        ]
        return rescaled_inputs


class attention(nn.Module):
    def __init__(self):
        super(attention, self).__init__()
        self.conv1x1_list = nn.ModuleList([
            # w-48 32->48 C->(48,96,192,384)
            nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding=0)
            for in_channels in [32, 64, 128, 256]
        ])
        self.bn_list = nn.ModuleList([
            nn.BatchNorm2d(32) for _ in range(4)
        ])
        self.relu = nn.ReLU(inplace=True)
        # self.up = F.interpolate(size=(128, 128), mode='nearest')
        self.conv1x1_output = nn.Conv2d(128, 4, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(4)
        self.relu2 = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.fc = nn.Linear(1, 1)

    def forward(self, inputs):
        if len(inputs) != 4:
            raise ValueError("The length of inputs should be 4.")
        # 对输入的各个列表元素进行通道映射，BN，ReLU
        conv1x1_inputs = []
        for i, input in enumerate(inputs):
            conv1x1_input = self.conv1x1_list[i](input)
            bn_input = self.bn_list[i](conv1x1_input)
            relu_input = self.relu(bn_input)
            conv1x1_inputs.append(relu_input)
        # 上采样操作
        up_inputs = [F.interpolate(input, size=(160, 160), mode='nearest') for input in conv1x1_inputs]
        # 按通道相加得到一个 (128, 128, 32) 的特征图
        # added_inputs = torch.stack(up_inputs).sum(dim=0)
        concat_input = torch.concat(up_inputs, dim=1)
        # 1x1 卷积映射为 (128, 128, 4)，BN，ReLU
        conv1x1_output = self.conv1x1_output(concat_input)
        bn_output = self.bn2(conv1x1_output)
        relu_output = self.relu2(bn_output)
        # 每个通道进行全局池化，sigmoid，仿射变换
        outputs = []
        for i in range(4):
            channel_output = relu_output[:, i, :, :]
            global_pool_output = self.global_pool(channel_output)
            sigmoid_output = self.sigmoid(global_pool_output)
            # softmax_output = self.softmax(global_pool_output)
            flatten_output = sigmoid_output.view(sigmoid_output.size(0), -1)
            affine_output = self.fc(flatten_output)
            affine_output = affine_output.unsqueeze(2).unsqueeze(3)
            outputs.append(affine_output)
        # 将输出与输入分别相乘
        multiplied_outputs = []
        for i in range(4):
            multiplied_outputs.append(torch.mul(outputs[i], inputs[i]))
        return multiplied_outputs