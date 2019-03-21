import torch
import torch.nn as nn
# import torch.nn.functional as F
from base import BaseModel
from model import common
from model.models import MSMNetModel, AttentionModel


class AMSMNetModel(BaseModel):
    def __init__(self, conv=common.default_conv, **kwargs,):
        super(AMSMNetModel, self).__init__()
        self.msmnet_model = MSMNetModel(conv, **kwargs)
        self.attention_model = AttentionModel(conv, **kwargs)
    
    def forward(self, x_scale1, x_scale2, x_scale3):
        ms_output = self.msmnet_model(x_scale1, x_scale2, x_scale3)
        attention_output = self.attention_model(x_scale1)
        output = torch.mul(ms_output, attention_output)
        return output