import torch
import torch.nn as nn
from base import BaseModel
from model import common
from model.models import MSMNetModel, AttentionModel, FusionModel


class AMSMNetModel(BaseModel):
    def __init__(self, conv=common.default_conv, **kwargs,):
        super(AMSMNetModel, self).__init__()
        num_resblocks = kwargs['num_resblocks']
        self.msmnet_model = MSMNetModel(conv, **kwargs)
        self.attention_model = AttentionModel(conv, **kwargs)
        self.fusion_model = FusionModel(conv, **kwargs)
    
    def forward(self, x_scale1, x_scale2, x_scale3):
        ms_output = self.msmnet_model(x_scale1, x_scale2, x_scale3)
        attention_output = self.attention_model(x_scale1)
        output = torch.cat((ms_output, attention_output), dim=1)
        output = self.fusion_model(output)

        return output
