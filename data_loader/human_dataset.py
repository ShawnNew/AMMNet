"""
    dataset create
Author: Zhengwei Li
Date  : 2018/12/24
"""
import cv2
import os
import random as r
import numpy as np
import torch
import torch.utils.data as data

def read_files(data_dir, dict):

    # image_name = os.path.join(data_dir, 'image', file_name['image'])
    # trimap_name = os.path.join(data_dir, 'trimap', file_name['trimap'])
    # alpha_name = os.path.join(data_dir, 'alpha', file_name['alpha'])

    image = cv2.imread(dict['img'])
    trimap = cv2.imread(dict['trimap'])
    alpha = cv2.imread(dict['gt'])
    img_b,img_g,img_r = cv2.split(image)
    trimap_b, trimap_g, trimap_r = cv2.split(trimap)
    alpha_b, alpha_g, alpha_r = cv2.split(alpha)
    image = cv2.merge([img_r,img_g,img_b])
    trimap = cv2.merge([trimap_r, trimap_g, trimap_b])
    alpha = cv2.merge([alpha_r, alpha_g, alpha_b])

    return image, trimap, alpha


def random_scale_and_creat_patch(image, trimap, alpha, patch_size):
    # random scale
    if r.random() < 0.5:
        h, w, c = image.shape
        scale = 0.75 + 0.5*r.random()
        image = cv2.resize(image, (int(w*scale),int(h*scale)), interpolation=cv2.INTER_CUBIC)
        trimap = cv2.resize(trimap, (int(w*scale),int(h*scale)), interpolation=cv2.INTER_NEAREST)
        alpha = cv2.resize(alpha, (int(w*scale),int(h*scale)), interpolation=cv2.INTER_CUBIC)    
    # creat patch
    # if r.random() < 0.5:
    #     h, w, c = image.shape
    #     if h > patch_size and w > patch_size:
    #         x = r.randrange(0, w - patch_size)
    #         y = r.randrange(0, h - patch_size)
    #         image = image[y:y + patch_size, x:x+patch_size, :]
    #         trimap = trimap[y:y + patch_size, x:x+patch_size, :]
    #         alpha = alpha[y:y+patch_size, x:x+patch_size, :]
    #     else:
    #         image = cv2.resize(image, (patch_size,patch_size), interpolation=cv2.INTER_CUBIC)
    #         trimap = cv2.resize(trimap, (patch_size,patch_size), interpolation=cv2.INTER_NEAREST)
    #         alpha = cv2.resize(alpha, (patch_size,patch_size), interpolation=cv2.INTER_CUBIC)
    # else:
    #     image = cv2.resize(image, (patch_size,patch_size), interpolation=cv2.INTER_CUBIC)
    #     trimap = cv2.resize(trimap, (patch_size,patch_size), interpolation=cv2.INTER_NEAREST)
    #     alpha = cv2.resize(alpha, (patch_size,patch_size), interpolation=cv2.INTER_CUBIC)
    image = cv2.resize(image, (patch_size,patch_size), interpolation=cv2.INTER_CUBIC)
    trimap = cv2.resize(trimap, (patch_size,patch_size), interpolation=cv2.INTER_NEAREST)
    alpha = cv2.resize(alpha, (patch_size,patch_size), interpolation=cv2.INTER_CUBIC)


    return image, trimap, alpha


def random_flip(image, trimap, alpha):

    if r.random() < 0.5:
        image = cv2.flip(image, 0)
        trimap = cv2.flip(trimap, 0)
        alpha = cv2.flip(alpha, 0)

    if r.random() < 0.5:
        image = cv2.flip(image, 1)
        trimap = cv2.flip(trimap, 1)
        alpha = cv2.flip(alpha, 1)
    return image, trimap, alpha
       
def np2Tensor(array):
    ts = (2, 0, 1)
    tensor = torch.FloatTensor(array.transpose(ts).astype(float))    
    return tensor

class human_matting_data(data.Dataset):
    """
    human_matting
    """

    def __init__(self, root_dir, patch_size):
        super().__init__()
        self.data_root = root_dir

        self.patch_size = patch_size
        imglist = os.path.join(self.data_root, 'train.txt')
        with open(imglist) as f:
            self.imgID = f.readlines()
        self.num = len(self.imgID)
        print("Dataset : file number %d"% self.num)




    def __getitem__(self, index):
        # read files
        img_path_ = self.imgID[index].strip().split(' ')[0]
        img_name_ = os.path.basename(img_path_)
        gt_path_ = self.imgID[index].strip().split(' ')[1]
        trimap_path_ = os.path.join(self.data_root, 'trimap-train', img_name_)
        image, trimap, alpha = read_files(self.data_root, {'img': img_path_,\
                                                            'gt': gt_path_,\
                                                            'trimap': trimap_path_})
        # NOTE ! ! !
        # trimap should be 3 classes : fg, bg. unsure
        trimap[trimap==0] = 0
        trimap[trimap==128] = 1
        trimap[trimap==255] = 2
        trimap[((trimap!=0) & (trimap!=1) & (trimap!=2))] = 1 

        # augmentation
        image, trimap, alpha = random_scale_and_creat_patch(image, trimap, alpha, self.patch_size)
        #image, trimap, alpha = random_flip(image, trimap, alpha)


        # normalize
        image = (image.astype(np.float32)) / 255.0
        alpha = alpha.astype(np.float32) / 255.0
        # to tensor
        image = np2Tensor(image)
        trimap = np2Tensor(trimap)
        alpha = np2Tensor(alpha)

        trimap = trimap[0,:,:].unsqueeze_(0)
        alpha = alpha[0,:,:].unsqueeze_(0)

        sample = {'image': image, 'trimap': trimap, 'gt': alpha}

        return sample

    def __len__(self):
        return self.num
