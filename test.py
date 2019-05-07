import os
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import get_instance
from torchvision.utils import make_grid, save_image
from PIL import Image
from scipy.misc import imresize
import numpy as np
import torch.nn.functional as F


def main(config, resume, device, output_path):
    # setup data_loader instances
    data_loader = getattr(module_data, config['adobe_data_loader']['type'])(
        config['adobe_data_loader']['args']['data_dir'],
        batch_size=32,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=4
    )
    #f = open(images)
    ## build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in ["mse", "sad"]]

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for _, sample_batched in enumerate(tqdm(data_loader)):
            img_scale1 = sample_batched['image'].to(device)
            img_scale2 = F.interpolate(img_scale1.clone(), scale_factor=0.5)
            img_scale3 = F.interpolate(img_scale1.clone(), scale_factor=0.25)
            original_size = sample_batched['size']

            gt = sample_batched['gt'].to(device)

            output = model(img_scale1, img_scale2, img_scale3)

           
            try:
                os.stat(output_path)
            except:
                os.mkdir(output_path)

            for i in range(len(img_scale1)):
                filename = os.path.join(output_path, os.path.basename(sample_batched['name'][i]))

                img_ = img_scale1[i].unsqueeze(0).cpu()
                alpha_ = output[i].unsqueeze(0).cpu()
                alpha_ = torch.where(
                    alpha_ < 0.1, 
                    torch.tensor(0.0), 
                    alpha_)
                matte_img_ = img_ * alpha_
                save_ = torch.cat((
                    img_, alpha_.repeat(1,3,1,1),
                    matte_img_ 
                ), dim=0)
                save_image(make_grid(save_, nrow=3), filename)

            # computing loss, metrics on test set
            loss = loss_fn(output, gt)
            batch_size = len(img_scale1)
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, gt) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({met.__name__ : total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    print(log)


if __name__ == '__main__':
    output_path = os.path.join(os.getcwd(), 'output')
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']

    main(config, args.resume, args.device, output_path)
