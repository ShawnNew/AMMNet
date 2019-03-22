import torch
import torch.nn as nn
from base import BaseModel
from model import common
from model.models import MSMNetModel, AttentionModel


class AMSMNetModel(BaseModel):
    def __init__(self, conv=common.default_conv, **kwargs,):
        super(AMSMNetModel, self).__init__()
        self.msmnet_model = MSMNetModel(conv, **kwargs)
        self.attention_model = AttentionModel(conv, **kwargs)
        fusion_model = nn.Sequential(
            common.BasicBlock(conv, 2, 64, 3),
            common.ResBlock(conv, 64, 3),
            common.BasicBlock(conv, 64, 1, 1, act=nn.Sigmoid())
        )
        self.fusion_model = fusion_model
    
    def forward(self, x_scale1, x_scale2, x_scale3):
        ms_output = self.msmnet_model(x_scale1, x_scale2, x_scale3)
        attention_output = self.attention_model(x_scale1)
        # output = torch.mul(ms_output, attention_output)
        #output = ms_output + attention_output
        output = torch.cat((ms_output, attention_output), dim=1)
        output = self.fusion_model(output)

        return output
