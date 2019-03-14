import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torchvision.models as models


def mse_loss(output, target):
    loss_ = F.mse_loss(output, target)
    return loss_

def smooth_l1_loss(pred, target):
    loss_ = F._smooth_l1_loss(pred, target)
    return loss_

def gradient_loss(pred, gradient, target):
    diff_ = (pred - target)
    loss_ = torch.mean((diff_**2) * gradient)
    return loss_

class VGG16ContentModel(nn.Module):
    def __init__(self, **kwargs):
        super(VGG16ContentModel, self).__init__()
        idx_feature_map = kwargs['feature_map']
        pretrained_model = getattr(models, kwargs['model'])(pretrained=True).features
        # get certain layers to construct content-loss module
        model = pretrained_model[:idx_feature_map+1] # new content Sequential module network
        self.model = model

    def forward(self, pred, gt):
        content_pred = self.model(pred)
        content_gt = self.model(gt)
        return F.mse_loss(content_pred, content_gt)


