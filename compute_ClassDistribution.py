import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
import ttach as tta
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
import _init_paths
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
import time
from PIL import Image
import json
from os.path import join
import torch.nn as nn

import itertools
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = '/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes'
DATA_LIST_PATH = '/home/xiaoqiguo2/MetaCorrection/datasets/cityscapes_list/val.txt'
SAVE_PATH = '/home/xiaoqiguo2/scratch/MetaCorrection/result/cityscapes'

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'val'

MODEL = 'DeeplabMulti'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def fast_hist(a, n):
    ka = (a >= 0) & (a < n)
    return np.bincount(a[ka], minlength=n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_CD(gt_dir, pred_dir, devkit_dir='/home/xiaoqiguo2/MetaCorrection/datasets/cityscapes_list'):
    """
    Compute IoU given the predicted colorized images and 
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
      info = json.load(fp)
    num_classes = np.int(info['classes'])
    print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)

    image_path_list = join(devkit_dir, 'train.txt')
    label_path_list = join(devkit_dir, 'train_label.txt')
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]

    CM = np.zeros(19)
    for ind in range(len(pred_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        CM += fast_hist(pred.flatten(), 19)
    return CM

if __name__ == '__main__':
    gt_dir ='/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes/train_label'
    pred_dir ='/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes/pseudo_bapa/'
    Class_dist = compute_CD(gt_dir, pred_dir)
    Class_dist_norm = Class_dist/(np.sum(Class_dist)+10e-10)
    np.save("../ClassDist/ClassDist_bapa.npy", Class_dist_norm)
    print(Class_dist, Class_dist_norm)