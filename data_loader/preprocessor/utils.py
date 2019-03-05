import os
import h5py
import cv2 as cv
import numpy as np
import random
from scipy import misc as misc
import math
import pdb

def writeH5Files(samples_array, file_path):
    """
    Write the formatted data into hdf5 file from img_array
    """
    dir_, _ = os.path.split(file_path)
    if not os.path.exists(dir_): os.mkdir(dir_)
    hdf_file = h5py.File(file_path, 'w')
    img = samples_array[:,:3,:,:]
    img_scale1 = scale_by_factor(img, 0.5)
    img_scale2 = scale_by_factor(img, 0.25)
    tri_map = np.expand_dims(samples_array[:,3,:,:], axis=1)
    gt = np.expand_dims(samples_array[:, 4, :, :], axis=1)
    #fg = samples_array[:, 5:8, :, :]
    #bg = samples_array[:, 8:11, :, :]
    gradient = np.expand_dims(samples_array[:, 11, :, :], axis=1)
    #roughness = np.expand_dims(samples_array[:, 12, :, :], axis=1)
    tri_map_original = np.expand_dims(samples_array[:,13, :, :], axis=1)
    hdf_file['img'] = img
    hdf_file['img-scale1'] = img_scale1
    hdf_file['img-scale2'] = img_scale2
    hdf_file['tri-map'] = tri_map
    hdf_file['gt'] = gt
    #hdf_file['fg'] = fg
    #hdf_file['bg'] = bg
    hdf_file['gradient'] = gradient
    #hdf_file['roughness'] = roughness
    hdf_file['tri-map-origin'] = tri_map_original
    hdf_file.flush()
    hdf_file.close()

def scale_by_factor(data, scale):
    res = np.zeros((data.shape[0], data.shape[1], int(data.shape[2]*scale), int(data.shape[3]*scale)))
    for i in range(len(data)):
        img = data[i]
        img = np.transpose(img, (1,2,0))
        scale_ = cv.resize(img, (0,0), fx=scale, fy=scale)
        scale_ = np.transpose(scale_, (2,0,1))
        res[i] = scale_
    return res

def writeH5TxtFile(_dir):
    """
    Generate a txt file that list all the HDF5 files of the dataset.
    Param:
        out_dir: the output directory
        h5dir:   the hdf5 dataset directory
    """
    for file_ in os.listdir(_dir):
        if os.path.isdir(os.path.join(_dir, file_)):
            txt_filename = file_ + ".txt"
            txt_path = os.path.join(_dir, txt_filename)
            with open(txt_path, 'w') as f:
                sub_dir = os.path.join(_dir, file_)
                for i in os.listdir(sub_dir):
                    if i.endswith(".h5"):
                        path = os.path.abspath(sub_dir)
                        path = os.path.join(path, i)
                        line_ = path + "\n"
                        f.write(line_)


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


def batch_resize_by_scale(img, scale, channels):
    '''
    :param img: The input image, should be shape like [:,:,channels]
    :param deter_h: The picture height as you wish to resize to
    :param deter_w: The picture width as you wish to resize to
    :return: A vector with shape [deter_h, deter_w, channels]
    '''
    shape_ = img.shape
    image = np.zeros([shape_[0]*int(scale), shape_[1]*int(scale), channels])
    # try:
    image[:, :, :3] = cv.resize(img[:, :, :3],
                                None, fx=scale,
                                fy=scale,
                                interpolation=cv.INTER_CUBIC).astype(np.float64)
    image[:, :, 3] = cv.resize(img[:, :, 3],
                               None, fx=scale,
                               fy=scale,
                               interpolation=cv.INTER_CUBIC).astype(np.float64)
    image[:, :, 4] = cv.resize(img[:, :, 4],
                               None, fx=scale,
                               fy=scale,
                               interpolation=cv.INTER_CUBIC).astype(np.float64)
    image[:, :, 5:8] = cv.resize(img[:, :, 5:8],
                                 None, fx=scale,
                                 fy=scale,
                                 interpolation=cv.INTER_CUBIC).astype(np.float64)
    image[:, :, 8:11] = cv.resize(img[:, :, 8:11],
                                  None, fx=scale,
                                  fy=scale,
                                  interpolation=cv.INTER_CUBIC).astype(np.float64)
    image[:, :, 11] = cv.resize(img[:, :, 11],
                                None, fx=scale,
                                fy=scale,
                                interpolation=cv.INTER_CUBIC).astype(np.float64)
    image[:, :, 12] = cv.resize(img[:, :, 12],
                                None, fx=scale,
                                fy=scale,
                                interpolation=cv.INTER_CUBIC).astype(np.float64)
    image[:,:,13] = cv.resize(img[:,:,13],
                              None, fx=scale,
                              fy=scale,
                              interpolation=cv.INTER_CUBIC).astype(np.float64)
    return image

def batch_resize(img, deter_h, deter_w, channels):
    '''
    :param img: The input image, should be shape like [:,:,channels]
    :param deter_h: The picture height as you wish to resize to
    :param deter_w: The picture width as you wish to resize to
    :return: A vector with shape [deter_h, deter_w, channels]
    '''
    image = np.zeros([deter_h, deter_w, channels])
    # try:
    image[:, :, :3] = cv.resize(img[:, :, :3],
                                (deter_w, deter_h),
                                interpolation=cv.INTER_CUBIC).astype(np.float64)
    image[:, :, 3] = cv.resize(img[:, :, 3],
                               (deter_w, deter_h),
                               interpolation=cv.INTER_CUBIC).astype(np.float64)
    image[:, :, 4] = cv.resize(img[:, :, 4],
                               (deter_w, deter_h),
                               interpolation=cv.INTER_CUBIC).astype(np.float64)
    image[:, :, 5:8] = cv.resize(img[:, :, 5:8],
                                 (deter_w, deter_h),
                                 interpolation=cv.INTER_CUBIC).astype(np.float64)
    image[:, :, 8:11] = cv.resize(img[:, :, 8:11],
                                  (deter_w, deter_h),
                                  interpolation=cv.INTER_CUBIC).astype(np.float64)
    image[:, :, 11] = cv.resize(img[:, :, 11],
                                (deter_w, deter_h),
                                interpolation=cv.INTER_CUBIC).astype(np.float64)
    image[:, :, 12] = cv.resize(img[:, :, 12],
                                (deter_w, deter_h),
                                interpolation=cv.INTER_CUBIC).astype(np.float64)
    image[:, :, 13] = cv.resize(img[:, :, 13],
                                (deter_w, deter_h),
                                interpolation=cv.INTER_CUBIC).astype(np.float64)
    return image



