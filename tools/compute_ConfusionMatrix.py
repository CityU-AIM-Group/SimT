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
from model.deeplab_multi import DeeplabMulti, Shallow_layers, Deep_layers
from model.deeplab_vgg import DeeplabVGG
from model.meta_deeplab_multi import Res_Deeplab
from model.deeplab_dsp import Res_Deeplab_DSP
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

def fast_hist(a, b, n33, n19):
    ka = (a >= 0) & (a < n33)
    return np.bincount(n19 * a[ka].astype(int) + b[ka], minlength=n33 * n19).reshape(n33, n19)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_CM(gt_dir, pred_dir, devkit_dir='/home/xiaoqiguo2/MetaCorrection/datasets/cityscapes_list'):
    """
    Compute IoU given the predicted colorized images and 
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
      info = json.load(fp)
    num_classes = np.int(info['classes'])
    print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train_1'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'train.txt')
    label_path_list = join(devkit_dir, 'train_label.txt')
    # image_path_list = join(devkit_dir, 'val.txt')
    # label_path_list = join(devkit_dir, 'label.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]

    CM = np.zeros((34,19))
    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        label = label_mapping(label, mapping)
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue
        CM += fast_hist(label.flatten(), pred.flatten(), 34, 19)
    return CM

def plot_NTM(trans_mat, normalize=True, title='NTM1', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(trans_mat, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    thresh = trans_mat.max() / 2.
    for i, j in itertools.product(range(trans_mat.shape[0]), range(trans_mat.shape[1])):
        num = '{:.3f}'.format(trans_mat[i, j]) if normalize else int(trans_mat[i, j])
        plt.text(j, i, num,
                 fontsize=2, 
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if np.float(num) > thresh else "black")
    plt.savefig('/home/xiaoqiguo2/MetaCorrection/'+title+'.png', transparent=True, dpi=600)


if __name__ == '__main__':
    gt_dir ='/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes/train_label'
    # gt_dir ='/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes/label'
    pred_dir ='/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes/pseudo_adaptsegnet'
    Confusion_matrix = compute_CM(gt_dir, pred_dir)
    Confusion_matrix_norm = Confusion_matrix/(np.sum(Confusion_matrix, axis=1, keepdims=True)+10e-6)
    np.save("../CM.npy", Confusion_matrix)
    np.save("../CM_norm.npy", Confusion_matrix_norm)
    plot_NTM(Confusion_matrix_norm, normalize=True, title='Training_CM_Adapt', cmap=plt.cm.Blues)
    # plot_NTM(Confusion_matrix_norm, normalize=True, title='Testing_CM', cmap=plt.cm.Blues)
    # print(Confusion_matrix)
    print(Confusion_matrix_norm)
    print(np.mean(np.diag(Confusion_matrix_norm[:19,:])))