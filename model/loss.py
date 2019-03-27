import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torchvision.models as models


def mse_loss(output, target):
    loss_ = F.mse_loss(output, target)
    return loss_

def smooth_l1_loss(pred, target):
    loss_ = F.smooth_l1_loss(pred, target)
    return loss_

def gradient_loss(pred, gradient, target):
    diff_ = (pred - target)
    loss_ = torch.mean((diff_**2) * gradient)
    return loss_

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ct, ht, wt = target.size()
    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
            input, target, weight=weight, size_average=size_average, ignore_index=250
        )
    return loss

class VGG16ContentModel(nn.Module):
    def __init__(self, **kwargs):
        super(VGG16ContentModel, self).__init__()
        pretrained_model = getattr(models, kwargs['model'])(pretrained=True).features
        self.weights = kwargs['weights']
        # get certain layers to construct content-loss module
        self.conv1= pretrained_model[:2]  # relu 1
        self.conv2 = pretrained_model[2:5] # maxpool 1
        self.conv3 = pretrained_model[5:7]
        self.conv4 = pretrained_model[7:10] # maxpool 2

    def forward(self, pred, gt):
        content_pred_conv1 = self.conv1(pred)
        content_gt_conv1 = self.conv1(gt)
        content_pred_conv2 = self.conv2(pred)
        content_gt_conv2 = self.conv2(gt)
        content_pred_conv3 = self.conv3(pred)
        content_gt_conv3 = self.conv3(gt)
        content_pred_conv4 = self.conv4(pred)
        content_gt_conv4 = self.conv4(gt)
        loss = self.weights[0] * F.mse_loss(content_pred_conv1, content_gt_conv1) +\
                self.weights[1] * F.mse_loss(content_pred_conv2, content_gt_conv2) +\
                self.weights[2] * F.mse_loss(content_pred_conv3, content_gt_conv3) +\
                self.weights[3] * F.mse_loss(content_pred_conv4, content_gt_conv4)
        return loss
