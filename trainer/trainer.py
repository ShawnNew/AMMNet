import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
import torch.nn.functional as F


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, content_loss, metrics, optimizer, resume, finetune, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, content_loss, metrics, optimizer, resume, finetune, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = self.config['trainer']['log_step']
        self.alpha_loss_weight = self.config['alpha_loss_weight']
        self.comp_loss_weight = self.config['comp_loss_weight']
        self.content_loss_weight = self.config['content_loss_weight']
        self.regularization = self.config['regularization']

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        dict_ = {}
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            dict_[f'{metric.__name__}'] = acc_metrics[i]
            # self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        self.writer.add_scalars('metrics', dict_)
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
    
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, sample_batched in enumerate(self.data_loader):
            img_scale1 = sample_batched['image'].to(self.device)
            img_scale2 = F.interpolate(img_scale1.clone(), scale_factor=0.5)
            img_scale3 = F.interpolate(img_scale1.clone(), scale_factor=0.25)

            gt = sample_batched['gt'].to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(img_scale1, img_scale2, img_scale3)
            ## content loss
            pred_object = img_scale1 * output
            gt_object = img_scale1 * gt
            content_loss_ = self.content_loss(pred_object, gt_object) *\
                            self.content_loss_weight
            ## comp loss
            fg = img_scale1 * gt
            bg = img_scale1 * (1-gt)
            color_pred = output * fg + (1-output) * bg
            comp_loss_ = self.loss(color_pred, img_scale1) * self.comp_loss_weight

            ## overall loss
            alpha_loss_ = self.loss(output, gt) * self.alpha_loss_weight

            ## l2 regularization
            reg_loss = None
            for param in self.model.parameters():
                if reg_loss is None:
                    reg_loss = 0.5 * torch.sum(param**2)
                else:
                    reg_loss = reg_loss + 0.5 * param.norm(2)**2
            
            loss = alpha_loss_ + comp_loss_ + content_loss_ +\
                    self.regularization * reg_loss
            
            ## backprop
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            self.writer.add_scalars('training-loss', {'alpha-loss':alpha_loss_.item(),
                                                      'comp-loss':comp_loss_.item(),
                                                      'content-loss':content_loss_.item()})
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, gt)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                self.writer.add_image('input', make_grid(img_scale1[:2].cpu(), nrow=2, normalize=True))
                self.writer.add_image('gt', make_grid(gt[:2].cpu(), nrow=2, normalize=True))
                self.writer.add_image('output', make_grid(output[:2].cpu(), nrow=2, normalize=True))
                

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        #import pdb
        #pdb.set_trace()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        # TODO: implement metrics
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(self.valid_data_loader):
                img_scale1 = sample_batched['image'].to(self.device)
                img_scale2 = F.interpolate(img_scale1.clone(), scale_factor=0.5)
                img_scale3 = F.interpolate(img_scale1.clone(), scale_factor=0.25)
 
                gt = sample_batched['gt'].to(self.device)

                output = self.model(img_scale1, img_scale2, img_scale3)
                loss = self.loss(output, gt)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, gt)
                self.writer.add_image('input', make_grid(img_scale1[:2].cpu(), nrow=2, normalize=True))
                self.writer.add_image('gt', make_grid(gt[:2].cpu(), nrow=2, normalize=True))
                self.writer.add_image('output', make_grid(output[:2].cpu(), nrow=2, normalize=True))

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
