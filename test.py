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
import cv2
import time
from utils.visualization import decode_segmap


def main(config, args):
    # output_path = os.path.join(os.getcwd(), 'output-human')
    current_dir = os.getcwd()
    output_path = os.path.join(current_dir, 'output-benchmark')
    try:
        os.stat(output_path)
    except:
        os.mkdir(output_path)
    # test_list_file = args.testList
    # test_list = []
    # test_dir = os.path.split(test_list_file)[0]
    # test_dir = os.path.join(test_dir, 'pic')
    # with open(test_list_file, 'r') as f:
    #    test_list = f.readlines()
            
    # setup data_loader instances
    data_loader = getattr(module_data, config['alphamatting_data_loader']['type'])(
        "/public/Datasets/alphamatting-dataset/",
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
    # loss_fn = getattr(module_loss, config['loss'])
    # metric_fns = [getattr(module_metric, met) for met in ["mse", "sad"]]

    # load state dict
    checkpoint = torch.load(args.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # total_loss = 0.0
    # total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, sample_batched in enumerate(data_loader):
        # for item in test_list:
            # name_ = item.strip()
            # img_path = os.path.join(test_dir, name_)
            #img_path = sample.strip().split(' ')[0]
            #name_ = os.path.basename(img_path)
            name_ = os.path.basename(sample_batched['name'][0])
            output_file_path = os.path.join(output_path, name_)
            # img = Image.open(img_path)
            # original_size = (img.height, img.width)
            # img_scale1 = torch.from_numpy(
            #    np.transpose(imresize(img, (320, 320)), (2,0,1)) / 255.
            #    ).type(torch.FloatTensor).unsqueeze(0)
            # img_scale1 = img_scale1.to(device)
            # img_scale2 = F.interpolate(img_scale1.clone(), scale_factor=0.5)
            # img_scale3 = F.interpolate(img_scale1.clone(), scale_factor=0.25)
            img_scale1 = sample_batched['image'].to(device)
            img_scale2 = F.interpolate(img_scale1.clone(), scale_factor=0.5)
            img_scale3 = F.interpolate(img_scale1.clone(), scale_factor=0.25)
            original_size = sample_batched['size']
            # gt = sample_batched['gt'].to(device)
            t0 = time.time()
            output = model(img_scale1, img_scale2, img_scale3)
            print(time.time() - t0)
            # pred = output.data.max(1)[1].cpu().numpy()
            # decoded = decode_segmap(pred, 3)
            # decoded = np.transpose(decoded[0], (1,2,0))
            # decoded = cv2.resize(decoded, (original_size[1], original_size[0]),\
            #     interpolation=cv2.INTER_CUBIC)
            # output_file_path = os.path.join(output_path, \
            #                 os.path.basename(sample_batched['name'][0]))
            alpha_pred = output[0,0,:,:].data.cpu().numpy()
            alpha_pred = np.where(
               alpha_pred < 0.1, 
               0.,
               alpha_pred
            )
            alpha_pred = np.where(
               alpha_pred > 0.9,
               1.,
               alpha_pred
            )
            alpha_pred = cv2.resize(alpha_pred, (original_size[1], original_size[0]),\
                interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(output_file_path, alpha_pred*255.)
           
            

            # alpha_pred = output[0,0,:,:].cpu().data.numpy()
            #alpha_pred = np.where(
            #    alpha_pred < 0.15, 
            #    0.,
            #    alpha_pred
            #)
            #alpha_pred = np.where(
            #    alpha_pred > 0.95,
            #    1.,
            #    alpha_pred
            #)
            # output_file_path = os.path.join(output_path, \
            #                     os.path.basename(sample_batched['name'][0]))
            # alpha_pred = cv2.resize(alpha_pred, (original_size[1], original_size[0]),\
            #     interpolation=cv2.INTER_CUBIC)
            # cv2.imwrite(output_file_path, alpha_pred*255.)

            # for i in range(len(img_scale1)):
            #     filename = os.path.join(output_path, os.path.basename(sample_batched['name'][i]))
            #     size_ = (original_size[1][i].item(), original_size[0][i].item())

            #     alpha_pred = output[i,0,:,:].cpu().data.numpy()
            #     alpha_pred = cv2.resize(alpha_pred, size_, interpolation=cv2.INTER_CUBIC)

            #     # alpha_pred = np.where(
            #     #     alpha_pred < 0.1, 
            #     #     0., 
            #     #     alpha_pred)
            #     # alpha_pred = np.where(
            #     #     alpha_pred > 0.95,
            #     #     1.,
            #     #     alpha_pred
            #     # )
            #     cv2.imwrite(filename, alpha_pred*255.)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('-t', '--testList', type=str, help='specify test dataset list.')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']

    main(config, args)
