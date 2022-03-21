import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image
import json
from os.path import join
import matplotlib.pyplot as plt
import torch.nn as nn

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = '/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes'
DATA_LIST_PATH = '../dataset/cityscapes_list/val.txt'
SAVE_PATH = '../result/cityscapes'

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
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32, 255, 255, 255]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)

def evaluate_simt(seg_model, pred_dir=None, devkit_dir='/home/xiaoqiguo2/SimT/dataset/cityscapes_list', post=False):
    """Create the model and start the evaluation process."""

    # if not os.path.exists(pred_dir):
    #     os.makedirs(pred_dir)
    device = torch.device("cuda")

    testloader = data.DataLoader(cityscapesDataSet(DATA_DIRECTORY, DATA_LIST_PATH, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=SET),
                                    batch_size=1, shuffle=False, pin_memory=True)
    testloader_640 = data.DataLoader(cityscapesDataSet(DATA_DIRECTORY, DATA_LIST_PATH, crop_size=(1280, 640), mean=IMG_MEAN, scale=False, mirror=False, set=SET),
                                    batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
    print('Evaluate for testing data')

    with open(join(devkit_dir, 'info.json'), 'r') as fp:
        info = json.load(fp)
    num_classes = np.int(info['classes'])
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))

    seg_model.eval()
    with torch.no_grad():
        for index, (batch, batch_640) in enumerate(zip(testloader, testloader_640)):
            image, _, name = batch
            image = image.to(device)

            image_640, _, name = batch_640
            image_640 = image_640.to(device)

            output1, output2 = seg_model(image)
            output = interp(output2[:,:num_classes,:,:]).cpu().data[0].numpy()
            del output1
            del output2

            output1, output2 = seg_model(image_640)
            output += interp(output2[:,:num_classes,:,:]).cpu().data[0].numpy()
            del output1
            del output2

            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2))
    
            gt_dir ='/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes/label'
            gt_path = '%s/%s' % (gt_dir, name[0].split('leftImg8bit')[0]+'gtFine_labelIds.png')

            label = np.array(Image.open(gt_path))
            label = label_mapping(label, mapping)
            if len(label.flatten()) != len(output.flatten()):
                print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
                continue
            hist += fast_hist(label.flatten(), output.flatten(), num_classes)

            # output_col = colorize_mask(np.uint8(output))
            # output = Image.fromarray(np.uint8(output))
            # name = name[0].split('/')[-1]
            # # pred_dir = '/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes/pseudo_sfdaseg_so'
            # pred_dir = '/home/xiaoqiguo2/SimT/result_SFDA'
            # output.save('%s/%s' % (pred_dir, name))
            # output_col.save('%s/%s_color.png' % (pred_dir, name.split('.')[0]))

    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    return round(np.nanmean(mIoUs) * 100, 2)


def evaluate_warmup(seg_model, pred_dir=None, devkit_dir='/home/xiaoqiguo2/SimT/dataset/cityscapes_list', post=False):
    """Create the model and start the evaluation process."""

    # if not os.path.exists(pred_dir):
    #     os.makedirs(pred_dir)
    device = torch.device("cuda")

    testloader = data.DataLoader(cityscapesDataSet(DATA_DIRECTORY, DATA_LIST_PATH, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=SET),
                                    batch_size=1, shuffle=False, pin_memory=True)
    testloader_640 = data.DataLoader(cityscapesDataSet(DATA_DIRECTORY, DATA_LIST_PATH, crop_size=(1280, 640), mean=IMG_MEAN, scale=False, mirror=False, set=SET),
                                    batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
    print('Evaluate for testing data')

    with open(join(devkit_dir, 'info.json'), 'r') as fp:
        info = json.load(fp)
    num_classes = np.int(info['classes'])
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))

    seg_model.eval()
    with torch.no_grad():
        for index, (batch, batch_640) in enumerate(zip(testloader, testloader_640)):
            image, _, name = batch
            image = image.to(device)

            image_640, _, name = batch_640
            image_640 = image_640.to(device)

            output1, output2 = seg_model(image)
            output = interp(output2).cpu().data[0].numpy()
            del output1
            del output2

            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2))
    
            gt_dir ='/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes/label'
            gt_path = '%s/%s' % (gt_dir, name[0].split('leftImg8bit')[0]+'gtFine_labelIds.png')

            label = np.array(Image.open(gt_path))
            label = label_mapping(label, mapping)
            if len(label.flatten()) != len(output.flatten()):
                print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
                continue
            hist += fast_hist(label.flatten(), output.flatten(), num_classes)

            # output_col = colorize_mask(np.uint8(output))
            # output = Image.fromarray(np.uint8(output))
            # name = name[0].split('/')[-1]
            # pred_dir = '/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes/pseudo_adapt_warmup'
            # output.save('%s/%s' % (pred_dir, name))
            # output_col.save('%s/%s_color.png' % (pred_dir, name.split('.')[0]))

    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    return round(np.nanmean(mIoUs) * 100, 2)