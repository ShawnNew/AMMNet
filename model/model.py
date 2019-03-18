import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model import common


class AMSMNetModel(BaseModel):
    def __init__(self, conv=common.default_conv, **kwargs,):
        super(AMSMNetModel, self).__init__()
        # -------------- Define multi-scale model architecture here ------------
        # scale = 3
        input_channles = kwargs['input_channels']
        num_resblocks = kwargs['num_resblocks']
        intermediate_channels = kwargs['intermediate_channels']
        kernel_size = kwargs['default_kernel_size']
        attention_input_channels = kwargs['attention_input_channels']
        dense_growth_rate = kwargs['dense_growth_rate']
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
                common.PreResBlock(conv, intermediate_channels, intermediate_channels, 5, bn=True, act=activation),
                common.ResBlock(conv, intermediate_channels, 5, bn=True, act=activation)
            )#,
            # nn.Sequential(
            #     common.ResBlock(conv, intermediate_channels, 5, bn=True, act=activation),
            #     common.ResBlock(conv, intermediate_channels, 5, bn=True, act=activation)
            # )
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

        # -------------- Define attention model here ----------------------
        self.attention_conv_head = nn.Sequential(
            common.BasicBlock(conv, attention_input_channels, intermediate_channels, kernel_size),
            common.BasicBlock(conv, intermediate_channels, intermediate_channels, kernel_size),
        )
        self.attention_maxpool = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.down_dense_1 = common.DenseBlock(3, intermediate_channels, bn_size=4, 
                                        growth_rate=dense_growth_rate, drop_rate=0.1)
        down_dense_1_output_channels = intermediate_channels + 3*dense_growth_rate
        self.down_dense_1_trans = common.Transition(down_dense_1_output_channels, 
                                                down_dense_1_output_channels // 2)
        self.down_dense_2 = common.DenseBlock(3, down_dense_1_output_channels // 2, 
                                        bn_size=4, growth_rate=dense_growth_rate, drop_rate=0.1)
        down_dense_2_output_channels = down_dense_1_output_channels // 2 + 3*dense_growth_rate
        self.down_dense_2_trans = common.Transition(down_dense_2_output_channels, 
                                            down_dense_2_output_channels // 2)
        self.bottom_dense = common.DenseBlock(3, down_dense_2_output_channels // 2, bn_size=4, 
                                        growth_rate=dense_growth_rate, drop_rate=0.1)
        bottom_dense_output_channels = down_dense_2_output_channels // 2 + 3*dense_growth_rate
        self.bottom_dense_upsample = common.Upsampler(conv, bottom_dense_output_channels)
        self.up_dense_2 = common.DenseBlock(3,
                                            bottom_dense_output_channels+down_dense_2_output_channels,
                                            bn_size=4, growth_rate=dense_growth_rate, drop_rate=0.1)
        
        up_dense_2_output_channels = bottom_dense_output_channels \
                        + down_dense_2_output_channels \
                            + 3*dense_growth_rate
        self.up_dense_2_upsample = common.Upsampler(conv, up_dense_2_output_channels)

        self.up_dense_1 = common.DenseBlock(3, up_dense_2_output_channels+down_dense_1_output_channels,
                                            bn_size=4, growth_rate=dense_growth_rate, drop_rate=0.1)
        up_dense_1_output_channels = up_dense_2_output_channels \
                        + down_dense_1_output_channels \
                            + 3*dense_growth_rate
        self.up_dense_1_upsample = common.Upsampler(conv, up_dense_1_output_channels)
        self.attention_conv_tail = nn.Sequential(
            common.BasicBlock(conv, up_dense_1_output_channels+intermediate_channels,
                            intermediate_channels, kernel_size),
            common.BasicBlock(conv, intermediate_channels, 1, kernel_size)
        )

    def forward(self, x_scale1, x_scale2):
        # ---------------- multi scale ------------------
        ## scale3(smallest)
        #x_scale3 = self.sub_mean(x_scale3)
        # x_scale3 = self.head(x_scale3)
        # x_scale3 = self.pre_process[2](x_scale3)

        # res_scale3 = self.body(x_scale3)
        # res_scale3 += x_scale3

        # x_scale3 = self.upsample(res_scale3)

        ## scale2
        #x_scale2 = self.sub_mean(x_scale2)
        x_scale2 = self.head(x_scale2)
        # concat upsampled scale
        # x_scale2 = torch.cat((x_scale3, x_scale2), dim=1)
        x_scale2 = self.pre_process[1](x_scale2)

        res_scale2 = self.body(x_scale2)
        res_scale2 += x_scale2

        x_scale2 = self.upsample(res_scale2)

        ## scale1
        #x_scale1 = self.sub_mean(x_scale1)
        x_scale1_temp = self.head(x_scale1)
        # concat upsampled scale
        x_scale1_temp = torch.cat((x_scale1_temp, x_scale2), dim=1)
        x_scale1_temp = self.pre_process[0](x_scale1_temp)

        res_scale1 = self.body(x_scale1_temp)
        res_scale1 += x_scale1_temp

        multi_output = self.tail(res_scale1)
        #x = self.add_mean(x)
        multi_output = self.output(multi_output)

        # ----------------- Attention -------------------
        attention = self.attention_conv_head(x_scale1)
        attention_maxpool, _ = self.attention_maxpool(attention)
        down_dense_1_ = self.down_dense_1(attention_maxpool)
        down_dense_1_trans = self.down_dense_1_trans(down_dense_1_)
        down_dense_2_ = self.down_dense_2(down_dense_1_trans)
        down_dense_2_trans = self.down_dense_2_trans(down_dense_2_)
        bottom_dense = self.bottom_dense(down_dense_2_trans)
        bottom_dense = self.bottom_dense_upsample(bottom_dense)
        up_dense_2_ = torch.cat((down_dense_2_, bottom_dense), dim=1)
        up_dense_2_ = self.up_dense_2(up_dense_2_)
        up_dense_2_ = self.up_dense_2_upsample(up_dense_2_)
        up_dense_1_ = torch.cat((up_dense_2_, down_dense_1_), dim=1)
        up_dense_1_ = self.up_dense_1(up_dense_1_)
        up_dense_1_ = self.up_dense_1_upsample(up_dense_1_)
        dense_output = torch.cat((up_dense_1_, attention), dim=1)
        attention_output_ = self.attention_conv_tail(dense_output)

        output = torch.mul(multi_output, attention_output_)


        return output
