import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


content_layers_default = ['conv_4']


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

def get_content_loss(content_net, device, pred, target, color_img, 
                     content_weight=1, content_layers=content_layers_default):
    # get content loss between pred and gt and return.
    cnn = copy.deepcopy(content_net)

    content_losses = []
    model = nn.Sequential() # new content Sequential module network
    model.to(device)

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(target).clone()
                content_loss = ContentLoss(pred, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)
        
        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)
    return content_losses                



class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # detach the target content from the content module
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        return self.loss

