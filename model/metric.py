import torch
import torch.nn.functional as F


def mse(output, target):
    with torch.no_grad():
        return F.mse_loss(output, target)


def sad(output, target):
    with torch.no_grad():
        return F.l1_loss(output, target, size_average=False)
