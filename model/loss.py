import torch.nn.functional as F


def mse_loss(output, target):
    return F.mse_loss(output, target)
