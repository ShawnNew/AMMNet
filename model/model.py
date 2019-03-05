import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model import common

def made_model():
    return AMSMNetModel()

class AMSMNetModel(BaseModel):
    def __init__(self, conv=common.default_conv):
        super(AMSMNetModel, self).__init__()
        # -------------- Define model architecture here ------------
        scale = 3
        input_channles = 3
        num_resblocks = 16
        intermediate_channels = 64
        kernel_size = 3
        activation = nn.ReLU(True)
        rgb_range = 255
        self.scale_idx = 0
        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        # head to read scaled image
        _head = [conv(input_channles, intermediate_channels, kernel_size)]

        # pre-process 2*Resblock each
        self.pre_process = nn.ModuleList([
            nn.Sequential(
                common.ResBlock(conv, intermediate_channels, 5, bn=True, act=activation),
                common.ResBlock(conv, intermediate_channels, 5, bn=True, act=activation)
            ) for _ in range(scale)
        ])

        # body 16*Resblocks each
        _body = [
            common.ResBlock(
                conv, intermediate_channels, kernel_size, bn=True, act=activation
            ) for _ in range(num_resblocks)
        ]
        _body.append(conv(intermediate_channels, intermediate_channels, kernel_size))

        # upsample to enlarge the scale
        self.upsample = nn.ModuleList([
            common.Upsampler(conv, s, intermediate_channels, bn=False, act=False) for s in range(scale)
        ])

        # tail to output prediction
        _tail = [conv(intermediate_channels, 1), kernel_size]

        self.head = nn.Sequential(*_head)
        self.body = nn.Sequential(*_body)
        self.tail = nn.Sequential(*_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        x = self.pre_process[self.scale_idx](x)

        res = self.body(x)
        res += x

        x = self.upsample[self.scale_idx](res)
        x = self.tail(x)
        x = self.add_mean(x)

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
    