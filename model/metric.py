import torch
import torch.nn.functional as F


def mse(output, target):
    with torch.no_grad():
        return F.mse_loss(output, target)
        #pred = torch.argmax(output, dim=1)
        #assert pred.shape[0] == len(target)
        #correct = 0
        #correct += torch.sum(pred == target).item()
    #return correct / len(target)

def sad(output, target):
    with torch.no_grad():
        return F.l1_loss(output, target) * output.shape[2] * output.shape[3]
    #     pred = torch.topk(output, k, dim=1)[1]
    #     assert pred.shape[0] == len(target)
    #     correct = 0
    #     for i in range(k):
    #         correct += torch.sum(pred[:, i] == target).item()
    # return correct / len(target)
