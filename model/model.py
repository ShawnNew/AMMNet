import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model import common


class AMSMNetModel(BaseModel):
    def __init__(self, conv=common.default_conv, **kwargs,):
        super(AMSMNetModel, self).__init__()
        # -------------- Define model architecture here ------------
        # scale = 3
        input_channles = kwargs['input_channels']
        num_resblocks = kwargs['num_resblocks']
        intermediate_channels = kwargs['intermediate_channels']
        kernel_size = kwargs['default_kernel_size']
        activation = nn.ReLU(True)
        rgb_range = kwargs['rgb_range']
        self.scale_idx = 0
        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        # head to read scaled image
        _head = common.BasicBlock(conv, input_channles, intermediate_channels, kernel_size)

        # pre-process 2*Resblock each
        self.pre_process = nn.ModuleList([
            nn.Sequential(
                common.PreResBlock(conv, 2*intermediate_channels, intermediate_channels, 5, bn=True, act=activation),
                common.ResBlock(conv, intermediate_channels, 5, bn=True, act=activation)
            ),
            nn.Sequential(
                common.PreResBlock(conv, 2*intermediate_channels, intermediate_channels, 5, bn=True, act=activation),
                common.ResBlock(conv, intermediate_channels, 5, bn=True, act=activation)
            ),
            nn.Sequential(
                common.ResBlock(conv, intermediate_channels, 5, bn=True, act=activation),
                common.ResBlock(conv, intermediate_channels, 5, bn=True, act=activation)
            )
        ])

        # body 16*Resblocks each
        _body = [
            common.ResBlock(
                conv, intermediate_channels, kernel_size, bn=True, act=activation
            ) for _ in range(num_resblocks)
        ]
        _body.append(conv(intermediate_channels, intermediate_channels, kernel_size))

        # upsample to enlarge the scale
        self.upsample = common.Upsampler(conv, intermediate_channels, bn=False, act=False)

        # tail to output prediction
        _tail = [common.BasicBlock(conv, intermediate_channels, 3, kernel_size)]

        _output = nn.ModuleList([
            conv(3, 1, 3),
            nn.Sigmoid()
        ])


        self.head = nn.Sequential(*_head)
        self.body = nn.Sequential(*_body)
        self.tail = nn.Sequential(*_tail)
        self.output = nn.Sequential(*_output)

    def forward(self, x_scale1, x_scale2, x_scale3):
        ## --------- scale3(smallest)
        #import pdb
        #pdb.set_trace()
        x_scale3 = self.sub_mean(x_scale3)
        x_scale3 = self.head(x_scale3)
        x_scale3 = self.pre_process[2](x_scale3)

        res_scale3 = self.body(x_scale3)
        res_scale3 += x_scale3

        x_scale3 = self.upsample(res_scale3)

        ## -------- scale2
        x_scale2 = self.sub_mean(x_scale2)
        x_scale2 = self.head(x_scale2)
        # concat upsampled scale
        x_scale2 = torch.cat((x_scale3, x_scale2), dim=1)
        x_scale2 = self.pre_process[1](x_scale2)

        res_scale2 = self.body(x_scale2)
        res_scale2 += x_scale2

        x_scale2 = self.upsample(res_scale2)

        ## -------- scale1
        x_scale1 = self.sub_mean(x_scale1)
        x_scale1 = self.head(x_scale1)
        # concat upsampled scale
        x_scale1 = torch.cat((x_scale1, x_scale2), dim=1)
        x_scale1 = self.pre_process[0](x_scale1)

        res_scale1 = self.body(x_scale1)
        res_scale1 += x_scale1

        x = self.tail(res_scale1)
        x = self.add_mean(x)
        x = self.output(x)

        return x
