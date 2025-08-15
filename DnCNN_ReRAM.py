import torch.nn as nn
from my_utils import Conv2d_fp8, Conv2d_fp8_hw, Conv2d_quant


class DnCNN(nn.Module):
    def __init__(self, channels,
                 num_of_layers=17,
                 qn_on: bool = False,
                 fp_on: int = 0,
                 weight_bit: int = 4,
                 output_bit: int = 8,
                 isint: int = 0,
                 clamp_std: int = 0,
                 quant_type: str ='None',
                 group_number: int =72,
                 left_shift_bit: int = 0):

        super(DnCNN, self).__init__()
        kernel_size = 3
        stride = 1
        padding = 1
        features = 64
        layers = []
        if (fp_on == 1):
            layers.append(
                Conv2d_fp8(in_channels=channels, out_channels=features, kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=group_number, bias=False))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(num_of_layers - 2):
                layers.append(
                    Conv2d_fp8(in_channels=features, out_channels=features, kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=group_number, bias=False))
                layers.append(nn.BatchNorm2d(features))
                layers.append(nn.ReLU(inplace=True))
            layers.append(
                Conv2d_fp8(in_channels=features, out_channels=channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=group_number, bias=False))
        elif (fp_on == 2):
            layers.append(
                Conv2d_fp8_hw(in_channels=channels, out_channels=features, kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=group_number, bias=False, quant_type=quant_type, group_number=group_number,
                              left_shift_bit=left_shift_bit))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(num_of_layers - 2):
                layers.append(
                    Conv2d_fp8_hw(in_channels=features, out_channels=features, kernel_size=kernel_size, stride=stride, padding=padding,
                                  groups=group_number, bias=False, quant_type=quant_type, group_number=group_number,
                                  left_shift_bit=left_shift_bit))
                layers.append(nn.BatchNorm2d(features))
                layers.append(nn.ReLU(inplace=True))
            layers.append(
                Conv2d_fp8_hw(in_channels=features, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=group_number, bias=False, quant_type=quant_type, group_number=group_number,
                              left_shift_bit=left_shift_bit))
        elif (qn_on):
            layers.append(Conv2d_quant(qn_on=qn_on, in_channels=channels, out_channels=features, kernel_size=kernel_size, stride=stride, padding=padding,
                                weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std, bias=False))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(num_of_layers - 2):
                layers.append(Conv2d_quant(qn_on=qn_on, in_channels=features, out_channels=features, kernel_size=kernel_size, stride=stride, padding=padding,
                                weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std, bias=False))
                layers.append(nn.BatchNorm2d(features))
                layers.append(nn.ReLU(inplace=True))
            layers.append(Conv2d_quant(qn_on=qn_on, in_channels=features, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                weight_bit=weight_bit, output_bit=output_bit, isint=isint, clamp_std=clamp_std, bias=False))
        else:
            layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(num_of_layers-2):
                layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
                layers.append(nn.BatchNorm2d(features))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        intermediates = []  # 用于保存每层的输出
        out = x
        for layer in self.dncnn:
            out = layer(out)
            # if isinstance(layer, nn.Conv2d):
            #     print((out<(out.min()/10)).sum()/out.numel())
            intermediates.append(out.clone())  # 保存每层的输出

        return out
