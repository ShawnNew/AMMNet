import os
import cv2 as cv
import numpy as np
import random
from scipy.misc import imresize
from PIL import Image
import math
# from torchvision import transforms
import torch


class MultiRescale(object):
    """MultiScale the input image in a sample by given scales.

    Args:
        scales_list (tuple or int): Desired output scale list. 
    """
    def __init__(self, scales_list):
        assert isinstance(scales_list, list)
        self.scales_list = scales_list

    def __call__(self, sample):
        height, width = sample['image'].height, sample['image'].width
        if (height%8 is not 0) or (width%8 is not 0):
            sample['image'] = Image.fromarray(imresize(sample['image'], (height//8*8, width//8*8)))
            sample['gt'] = Image.fromarray(imresize(sample['gt'], (height//8*8, width//8*8)))
            sample['trimap'] = Image.fromarray(imresize(sample['trimap'], (height//8*8, width//8*8)))
            sample['gradient'] = Image.fromarray(imresize(sample['gradient'], (height//8*8, width//8*8)))
        multi_scale_sample = {}
        multi_scale_sample['name'] = sample['name']
        multi_scale_sample['gt'] = sample['gt']
        multi_scale_sample['trimap'] = sample['trimap']
        multi_scale_sample['gradient'] = sample['gradient']
        i = 1
        for scale in self.scales_list:
            if scale == 1:
                img = sample['image']
            else:
                img = Image.fromarray(imresize(sample['image'], scale))
            img_name = 'image-scale' + str(i)
            multi_scale_sample[img_name] = img
            i += 1
        return multi_scale_sample



class RandomCrop(object):
    """Crop the images randomly in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """
    random_crop_list = [320, 480, 640]
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        random_crop_size = random.choice(self.random_crop_list)
        img, gt, trimap, grad = sample['image'], sample['gt'], sample['trimap'], sample['gradient']
        # crop along unknown region
        img_ = np.asarray(img)
        gt_ = np.asarray(gt)
        trimap_ = np.asarray(trimap)
        grad_ = np.asarray(grad)
        h_start = h_end = w_start = w_end = 0
        if (min(trimap_.shape) < random_crop_size):
            h_start = w_start = 0
            h_end = w_end = min(trimap_.shape)
        else:
            h_start, h_end, w_start, w_end = validUnknownRegion(trimap_, random_crop_size)
        img_ = img_[h_start:h_end, w_start:w_end, :]
        trimap_ = trimap_[h_start:h_end, w_start:w_end]
        gt_ = gt_[h_start:h_end, w_start:w_end]
        grad_ = grad_[h_start:h_end, w_start:w_end]
        
        # resize
        img = imresize(img_, self.output_size)
        trimap = imresize(trimap_, self.output_size)
        gt = imresize(gt_, self.output_size)
        grad = imresize(grad_, self.output_size)
        cropped_sample = {
            'name': sample['name'],
            'image': Image.fromarray(img), 
            'gt': Image.fromarray(gt),
            'trimap': Image.fromarray(trimap),
            'gradient': Image.fromarray(grad)
        }
        return cropped_sample


class MultiToTensor(object):
    def __call__(self, sample):
        gt, trimap, grad = sample['gt'], sample['trimap'], sample['gradient']
        img_scale1 = sample['image-scale1']
        img_scale2 = sample['image-scale2']
        img_scale3 = sample['image-scale3']
        img_scale1 = np.transpose(np.asarray(img_scale1), (2, 0, 1)) / 255.
        img_scale2 = np.transpose(np.asarray(img_scale2), (2, 0, 1)) / 255.
        img_scale3 = np.transpose(np.asarray(img_scale3), (2, 0, 1)) / 255.
        gt = np.expand_dims(np.asarray(gt), axis=0) / 255.
        trimap = np.expand_dims(np.asarray(trimap), axis=0) / 255.
        grad = np.expand_dims(np.asarray(grad), axis=0) / 255.
        return {
            'name': sample['name'],
            'image-scale1': torch.from_numpy(img_scale1).type(torch.FloatTensor),
            'image-scale2': torch.from_numpy(img_scale2).type(torch.FloatTensor),
            'image-scale3': torch.from_numpy(img_scale3).type(torch.FloatTensor),
            'gt': torch.from_numpy(gt).type(torch.FloatTensor),
            'trimap': torch.from_numpy(trimap).type(torch.FloatTensor),
            'gradient': torch.from_numpy(grad).type(torch.FloatTensor)
        }
        

def generate_gradient_map(grad, area=3):
    ## Generate gradient map based on computed gradient.
    # This function is used to count the gradient pixels passed a certain small area.
    # Parameters:
    #   grad: a gradient matrix
    #   area: small area to average
    # Output:
    #   grad_map
    num_pixel = int(area / 2)
    col_ = grad.shape[1]
    row_ = grad.shape[0] + 2*num_pixel
    new_row = np.zeros([num_pixel, col_], dtype=np.float32)
    new_col = np.zeros([row_, num_pixel], dtype=np.float32)
    result = np.zeros_like(grad)

    _tmp = np.r_[new_row, grad, new_row]
    _tmp = np.c_[new_col, _tmp, new_col]
    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            area_count = _tmp[i][j] + _tmp[i][j+1] + _tmp[i][j+2] +\
                        _tmp[i+1][j] + _tmp[i+1][j+1] + _tmp[i+1][j+2] +\
                        _tmp[i+2][j] + _tmp[i+2][j+1] + _tmp[i+2][j+2]
            result[i][j] = area_count / (area **2)
    return result

def getFileList(base, sub):
    """
    Get file list of a directory:
    Param:
        base: base directory
        sub: sub-directory name
    Return:
        a list of image file name
    """
    path = os.path.join(base, sub)
    files = os.listdir(path)
    fileList = []
    for f in files:
        if (os.path.isfile(path + '/' + f)):
            path_ = './' + sub
            path_ = os.path.join(path_, f)
            # add image file into list
            fileList.append(path_)
    return fileList


def candidateUnknownRegion(img):
    '''
    Propose a condidate of unknown region center randomly within the unknown area of img.
    param:
        img: trimap image
    return:
        an index for unknown region
    '''
    index = np.where(img == 128)
    idx = random.choice([j for j in range(len(index[0]))])
    return np.array(index)[:, idx]

def validUnknownRegion(img, output_size):
    """
    Check wether the candidate unknown region is valid and return the index.
    param:
        img:            trimap image
        output_size:    the desired output image size
    return:
        output the crop start and end index along h and w respectively.
    """
    h_start = h_end = w_start = w_end = 0
    cand = candidateUnknownRegion(img)
    shape_ = img.shape
    if (output_size == 320):
        h_start = max(cand[0]-159, 0)
        w_start = max(cand[1]-159, 0)
        if (h_start+320 > shape_[0]):
            h_start = shape_[0] - 320
        if (w_start+320 > shape_[1]):
            w_start = shape_[1] - 320
        h_end = h_start + 320
        w_end = w_start + 320
        return h_start, h_end, w_start, w_end
    elif (output_size == 480):
        h_start = max(cand[0]-239, 0)
        w_start = max(cand[1]-239, 0)
        if (h_start+480 > shape_[0]):
            h_start = shape_[0] - 480
        if (w_start+480 > shape_[1]):
            w_start = shape_[1] - 480
        h_end = h_start + 480
        w_end = w_start + 480
    elif (output_size == 640):
        h_start = max(cand[0]-319, 0)
        w_start = max(cand[1]-319, 0)
        if (h_start+640 > shape_[0]):
            h_start = shape_[0] - 640
        if (w_start+640 > shape_[1]):
            w_start = shape_[1] - 640
        h_end = h_start + 640
        w_end = w_start + 640
    return h_start, h_end, w_start, w_end
