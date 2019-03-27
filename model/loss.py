import torch.nn.functional as F


def mse_loss(output, target):
    return F.mse_loss(output, target)

class VGG16ContentModel(nn.Module):
    def __init__(self, **kwargs):
        super(VGG16ContentModel, self).__init__()
        idx_feature_map = kwargs['feature_map']
        pretrained_model = getattr(models, kwargs['model'])(pretrained=True).features
        # get certain layers to construct content-loss module
        #model = pretrained_model[:idx_feature_map+1] # new content Sequential module network
        self.conv1= pretrained_model[:1]
        self.conv2 = pretrained_model[1:3]
        self.conv3 = pretrained_model[3:6]
        self.conv4 = pretrained_model[6:8]
        #self.model = model

    def forward(self, pred, gt):
        #content_pred = self.model(pred)
        #content_gt = self.model(gt)
        content_pred_conv1 = self.conv1(pred)
        content_gt_conv1 = self.conv1(gt)
        content_pred_conv2 = self.conv2(pred)
        content_gt_conv2 = self.conv2(gt)
        content_pred_conv3 = self.conv3(pred)
        content_gt_conv3 = self.conv3(gt)
        content_pred_conv4 = self.conv4(pred)
        content_gt_conv4 = self.conv4(gt)
        loss = F.mse_loss(content_pred_conv1, content_gt_conv1)+F.mse_loss(content_pred_conv2, content_gt_conv2)+ \
               F.mse_loss(content_pred_conv3, content_gt_conv3)+F.mse_loss(content_pred_conv4, content_gt_conv4)
        #return F.mse_loss(content_pred, content_gt)
        return loss
