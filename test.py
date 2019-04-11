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
    data_loader = getattr(module_data, config['alphamatting_data_loader']['type'])(
        config['alphamatting_data_loader']['args']['data_dir'],
        batch_size=1,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=1
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
        for batch_idx, sample_batched in enumerate(tqdm(data_loader)):
            img_scale1 = sample_batched['image'].to(device)
            img_scale2 = F.interpolate(img_scale1.clone(), scale_factor=0.5)
            img_scale3 = F.interpolate(img_scale1.clone(), scale_factor=0.25)
            trimap = sample_batched['trimap']
            height, width = img_scale1.shape[2], img_scale2.shape[3]
            # gt = sample_batched['gt'].to(device)
            if img_scale1.shape[2] < 1000 and img_scale1.shape[3] < 1000:
                output = model(img_scale1, img_scale2, img_scale3)
                # trimap_scaled = trimap / 255.
                #masked_output = np.where(trimap==0, 0., output.cpu().numpy())
                #masked_output = np.where(trimap==255, 1., masked_output)
                #masked_output = torch.from_numpy(masked_output).type(torch.FloatTensor).to(device)
 
                img_path = sample_batched['name'][0]
                filename = os.path.basename(img_path)
                filename = os.path.join(output_path, filename)
                try:
                    os.stat(output_path)
                except:
                    os.mkdir(output_path)

                save_ = torch.cat((
                    img_scale1, 
                    output.repeat(1,3,1,1)), dim=0)
                save_image(make_grid(save_.cpu(), nrow=3), filename)
                # computing loss, metrics on test set
                # loss = loss_fn(output, gt)
                # batch_size = gt.shape[0]
                # total_loss += loss.item() * batch_size
                # for i, metric in enumerate(metric_fns):
                #     total_metrics[i] += metric(output, gt) * batch_size

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
