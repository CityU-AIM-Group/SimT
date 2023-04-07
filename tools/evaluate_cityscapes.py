import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
import statistics

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Deeplab
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
NUM_CLASSES = 16
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'val'

open_class = [9, 14, 16]
class_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18, 9, 14, 16]
class_mapping_inv = [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 9, 10, 11, 12, 17, 13, 18, 14, 15]

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



def evaluate_simt_unknown(seg_model, pred_dir=None, devkit_dir='/home/xiaoqiguo2/SimT_Plus/dataset/cityscapes_list', post=False):
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
    num_classes = np.int(info['closed classes'])
    open_classes = np.int(info['open classes'])
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes+1, num_classes+1))

    name_classes1 = name_classes.copy()
    for src, trg in enumerate(class_mapping_inv):
        name_classes[trg] = name_classes1[src]
    name_classes[num_classes] = 'unknown'
    name_classes = name_classes[:num_classes+1]

    seg_model.eval()
    with torch.no_grad():
        for index, (batch, batch_640) in enumerate(zip(testloader, testloader_640)):
            image, _, name = batch
            image = image.to(device)

            image_640, _, name = batch_640
            image_640 = image_640.to(device)

            # output1, output2 = seg_model(image)
            # output = interp(output2).cpu().data[0].numpy()
            # del output1
            # del output2

            output1, output2 = seg_model(image_640)
            output = interp(output2).cpu().data[0].numpy()
            del output1
            del output2

            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2))
            output1 = np.where(output >= num_classes, num_classes*np.ones_like(output), output)
            output = np.where(output == 255, output, output1)

            gt_dir ='/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes/label'
            gt_path = '%s/%s' % (gt_dir, name[0].split('leftImg8bit')[0]+'gtFine_labelIds.png')

            label = np.array(Image.open(gt_path))
            label = label_mapping(label, mapping)
            label1 = label.copy()
            for src, trg in enumerate(class_mapping_inv):
                label[label1 == src] = trg
            label1 = np.where(label >= num_classes, num_classes*np.ones_like(label), label)
            label = np.where(label == 255, label, label1)

            if len(label.flatten()) != len(output.flatten()):
                print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
                continue
            hist += fast_hist(label.flatten(), output.flatten(), num_classes+1)

            # output_col = colorize_mask(np.uint8(output))
            # output = Image.fromarray(np.uint8(output))
            # name = name[0].split('/')[-1]
            # # pred_dir = '/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes/pseudo_sfdaseg_so'
            # pred_dir = '/home/xiaoqiguo2/SimT_Plus/result_SFDA'
            # output.save('%s/%s' % (pred_dir, name))
            # output_col.save('%s/%s_color.png' % (pred_dir, name.split('.')[0]))

    mIoUs = per_class_iu(hist)

    for ind_class in range(num_classes+1):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU 16 classes: ' + str(round(np.nanmean(mIoUs[:num_classes]) * 100, 2)))
    print('===> mIoU unknown classe: ' + str(round(mIoUs[num_classes] * 100, 2)))
    print('===> mIoU 16+1 classes: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    print('===> mIoU HOS: ' + str(round(statistics.harmonic_mean([np.nanmean(mIoUs[:num_classes]), mIoUs[num_classes]+10e-6]) * 100, 2)))
    return round(statistics.harmonic_mean([np.nanmean(mIoUs[:num_classes]), mIoUs[num_classes]+10e-6]) * 100, 2)
    # return round(np.nanmean(mIoUs) * 100, 2)


def evaluate_simt_3unknown(seg_model, pred_dir=None, devkit_dir='/home/wzhou38/SimT_Plus/dataset/cityscapes_list', post=False, with_prior=False):
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
    close_classes = np.int(info['closed classes'])
    num_classes = np.int(info['closed classes']) + np.int(info['open classes'])
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    close_hist = np.zeros((close_classes, close_classes))
    hist = np.zeros((num_classes, num_classes))

    if with_prior:
        sprior = io.loadmat('../prior_array.mat') ##Please download this matrix from 'https://drive.google.com/file/d/1XJY1uiAlbxyoA-V4Mx8L7UgOlXRHvO0j/view?usp=sharing'
        prior_array = sprior["prior_array"].astype(np.float32)
        # index_close = class_mapping[:close_classes]
        # index_open = class_mapping[close_classes:]
        # prior_array = np.concatenate((prior_array[index_close], prior_array[index_open]), axis=0)
        prior_array = torch.from_numpy(prior_array).cuda()[8:10,:,:]
        prior_array = prior_array / torch.sum(prior_array, dim=0, keepdims=True)
        prior_veg = prior_array[0]
        prior_ter = prior_array[1]

    seg_model.eval()
    with torch.no_grad():
        for index, (batch, batch_640) in enumerate(zip(testloader, testloader_640)):
            # image, _, name = batch
            # image = image.to(device)

            image_640, _, name = batch_640
            image_640 = image_640.to(device)

            # output1, output2 = seg_model(image)
            # output = interp(output2).cpu().data[0].numpy()
            # del output1
            # del output2

            output1, output2 = seg_model(image_640)
            output = interp(output2)
            output_close = interp(output2[:,:close_classes,:,:]).cpu().data[0].numpy()
            del output1
            del output2

            if with_prior:
                output_spatial = torch.zeros_like(output)          
                output_spatial[:,8,:,:] = prior_veg * output[:,8,:,:]
                output_spatial[:,16,:,:] = prior_ter * output[:,16,:,:]
                output_spatial = torch.argmax(output_spatial, dim=1)

                output = torch.argmax(output, dim=1)
                mask = (output == 8) + (output == 16)
                output = torch.where(mask, output_spatial, output)
                output = output.cpu().data[0].numpy()
            else:
                output = output.cpu().data[0].numpy().transpose(1,2,0)
                output = np.asarray(np.argmax(output, axis=2))

            output1 = output.copy()
            for src, trg in enumerate(class_mapping):
                output[output1 == src] = trg

            output_close = output_close.transpose(1,2,0)
            output_close = np.asarray(np.argmax(output_close, axis=2))

            gt_dir ='/home/wzhou38/scratch/UDA_Natural/Cityscapes/label'
            gt_path = '%s/%s' % (gt_dir, name[0].split('leftImg8bit')[0]+'gtFine_labelIds.png')

            label = np.array(Image.open(gt_path))
            label = label_mapping(label, mapping)

            label_close = label.copy()
            for src, trg in enumerate(class_mapping_inv):
                label_close[label == src] = trg

            if len(label.flatten()) != len(output.flatten()):
                print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(output.flatten()), gt_imgs[ind], pred_imgs[ind]))
                continue
            hist += fast_hist(label.flatten(), output.flatten(), num_classes)
            close_hist += fast_hist(label_close.flatten(), output_close.flatten(), close_classes)

            # output_col = colorize_mask(np.uint8(output))
            # output = Image.fromarray(np.uint8(output))
            # name = name[0].split('/')[-1]
            # # pred_dir = '/home/wzhou38/scratch/UDA_Natural/Cityscapes/pseudo_sfdaseg_so'
            # pred_dir = '/home/wzhou38/SimT_Plus/result'
            # output.save('%s/%s' % (pred_dir, name))
            # output_col.save('%s/%s_color.png' % (pred_dir, name.split('.')[0]))

    mIoUs = per_class_iu(hist)
    close_mIoUs = per_class_iu(close_hist)

    # mIoUs16 = np.copy(mIoUs)
    # for idx, cl in enumerate(open_class):
    #     mIoUs16 = np.delete(mIoUs16, cl-idx, 0)
    # mIoUs16 = np.nanmean(mIoUs16)
    # mIoUs3 = (np.nanmean(mIoUs) * 19 - mIoUs16 * 16) / 3.

    mIoUs3 = np.nanmean(mIoUs[open_class])
    mIoUs16 = (np.nanmean(mIoUs) * num_classes - np.nanmean(mIoUs[open_class]) * len(open_class)) / (num_classes - len(open_class))

    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU close classes: ' + str(round(np.nanmean(close_mIoUs) * 100, 2)))
    print('===> mIoU 16 classes: ' + str(round(mIoUs16 * 100, 2)))
    print('===> mIoU 3 classes: ' + str(round(mIoUs3 * 100, 2)))
    print('===> mIoU 19 classes: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    print('===> mIoU HOS: ' + str(round(statistics.harmonic_mean([mIoUs16, mIoUs3+10e-6]) * 100, 2)))
    return round(statistics.harmonic_mean([mIoUs16, mIoUs3+10e-6]) * 100, 2)


'''def evaluate_simt_3unknown(seg_model, pred_dir=None, devkit_dir='/home/xiaoqiguo2/SimT_Plus/dataset/cityscapes_list', post=False):
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
    close_classes = np.int(info['closed classes'])
    num_classes = np.int(info['closed classes']) + np.int(info['open classes'])
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    close_hist = np.zeros((close_classes, close_classes))
    hist = np.zeros((num_classes, num_classes))

    seg_model.eval()
    with torch.no_grad():
        for index, (batch, batch_640) in enumerate(zip(testloader, testloader_640)):
            image, _, name = batch
            image = image.to(device)

            image_640, _, name = batch_640
            image_640 = image_640.to(device)

            # output1, output2 = seg_model(image)
            # output = interp(output2).cpu().data[0].numpy()
            # del output1
            # del output2

            output1, output2 = seg_model(image_640)
            output = interp(output2).cpu().data[0].numpy()
            output_close = interp(output2[:,:close_classes,:,:]).cpu().data[0].numpy()
            del output1
            del output2

            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2))
            output1 = output.copy()
            for src, trg in enumerate(class_mapping):
                output[output1 == src] = trg

            output_close = output_close.transpose(1,2,0)
            output_close = np.asarray(np.argmax(output_close, axis=2))

            gt_dir ='/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes/label'
            gt_path = '%s/%s' % (gt_dir, name[0].split('leftImg8bit')[0]+'gtFine_labelIds.png')

            label = np.array(Image.open(gt_path))
            label = label_mapping(label, mapping)

            label_close = label.copy()
            for src, trg in enumerate(class_mapping_inv):
                label_close[label == src] = trg

            if len(label.flatten()) != len(output.flatten()):
                print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
                continue
            hist += fast_hist(label.flatten(), output.flatten(), num_classes)
            close_hist += fast_hist(label_close.flatten(), output_close.flatten(), close_classes)

            # output_col = colorize_mask(np.uint8(output))
            # output = Image.fromarray(np.uint8(output))
            # name = name[0].split('/')[-1]
            # # pred_dir = '/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes/pseudo_sfdaseg_so'
            # pred_dir = '/home/xiaoqiguo2/SimT_Plus/result'
            # output.save('%s/%s' % (pred_dir, name))
            # output_col.save('%s/%s_color.png' % (pred_dir, name.split('.')[0]))

    mIoUs = per_class_iu(hist)
    close_mIoUs = per_class_iu(close_hist)

    # mIoUs16 = np.copy(mIoUs)
    # for idx, cl in enumerate(open_class):
    #     mIoUs16 = np.delete(mIoUs16, cl-idx, 0)
    # mIoUs16 = np.nanmean(mIoUs16)
    # mIoUs3 = (np.nanmean(mIoUs) * 19 - mIoUs16 * 16) / 3.

    mIoUs3 = np.nanmean(mIoUs[open_class])
    mIoUs16 = (np.nanmean(mIoUs) * num_classes - np.nanmean(mIoUs[open_class]) * len(open_class)) / (num_classes - len(open_class))

    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU close classes: ' + str(round(np.nanmean(close_mIoUs) * 100, 2)))
    print('===> mIoU 16 classes: ' + str(round(mIoUs16 * 100, 2)))
    print('===> mIoU 3 classes: ' + str(round(mIoUs3 * 100, 2)))
    print('===> mIoU 19 classes: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    print('===> mIoU HOS: ' + str(round(statistics.harmonic_mean([mIoUs16, mIoUs3+10e-6]) * 100, 2)))
    return round(statistics.harmonic_mean([mIoUs16, mIoUs3+10e-6]) * 100, 2)
'''

def evaluate_simt(seg_model, pred_dir=None, devkit_dir='/home/xiaoqiguo2/SimT_Plus/dataset/cityscapes_list', post=False):
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
    close_classes = np.int(info['closed classes'])
    num_classes = np.int(info['closed classes']) + np.int(info['open classes'])
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    close_hist = np.zeros((close_classes, close_classes))
    hist = np.zeros((num_classes, num_classes))

    seg_model.eval()
    with torch.no_grad():
        for index, (batch, batch_640) in enumerate(zip(testloader, testloader_640)):
            image, _, name = batch
            image = image.to(device)

            image_640, _, name = batch_640
            image_640 = image_640.to(device)

            # output1, output2 = seg_model(image)
            # output = interp(output2).cpu().data[0].numpy()
            # del output1
            # del output2

            output1, output2 = seg_model(image_640)
            output = interp(output2).cpu().data[0].numpy()
            output_close = interp(output2[:,:close_classes,:,:]).cpu().data[0].numpy()
            del output1
            del output2

            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2))
            output1 = output.copy()
            for src, trg in enumerate(class_mapping):
                output[output1 == src] = trg

            output_close = output_close.transpose(1,2,0)
            output_close = np.asarray(np.argmax(output_close, axis=2))

            gt_dir ='/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes/label'
            gt_path = '%s/%s' % (gt_dir, name[0].split('leftImg8bit')[0]+'gtFine_labelIds.png')

            label = np.array(Image.open(gt_path))
            label = label_mapping(label, mapping)

            label_close = label.copy()
            for src, trg in enumerate(class_mapping_inv):
                label_close[label == src] = trg

            if len(label.flatten()) != len(output.flatten()):
                print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
                continue
            hist += fast_hist(label.flatten(), output.flatten(), num_classes)
            close_hist += fast_hist(label_close.flatten(), output_close.flatten(), close_classes)

            # output_col = colorize_mask(np.uint8(output))
            # output = Image.fromarray(np.uint8(output))
            # name = name[0].split('/')[-1]
            # # pred_dir = '/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes/pseudo_sfdaseg_so'
            # pred_dir = '/home/xiaoqiguo2/SimT_Plus/result'
            # output.save('%s/%s' % (pred_dir, name))
            # output_col.save('%s/%s_color.png' % (pred_dir, name.split('.')[0]))

    mIoUs = per_class_iu(hist)
    close_mIoUs = per_class_iu(close_hist)

    # mIoUs16 = np.copy(mIoUs)
    # for idx, cl in enumerate(open_class):
    #     mIoUs16 = np.delete(mIoUs16, cl-idx, 0)
    # mIoUs16 = np.nanmean(mIoUs16)
    # mIoUs3 = (np.nanmean(mIoUs) * 19 - mIoUs16 * 16) / 3.

    mIoUs3 = np.nanmean(mIoUs[open_class])
    mIoUs16 = (np.nanmean(mIoUs) * num_classes - np.nanmean(mIoUs[open_class]) * len(open_class)) / (num_classes - len(open_class))

    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU close classes: ' + str(round(np.nanmean(close_mIoUs) * 100, 2)))
    print('===> mIoU 16 classes: ' + str(round(mIoUs16 * 100, 2)))
    print('===> mIoU 3 classes: ' + str(round(mIoUs3 * 100, 2)))
    print('===> mIoU 19 classes: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    return round(np.nanmean(close_mIoUs) * 100, 2)


def evaluate_warmup(seg_model, pred_dir=None, devkit_dir='/home/xiaoqiguo2/SimT_Plus/dataset/cityscapes_list', post=False):
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
    num_classes = np.int(info['closed classes']) + np.int(info['open classes'])
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

            # output1, output2 = seg_model(image)
            # output = interp(output2).cpu().data[0].numpy()
            # del output1
            # del output2

            output1, output2 = seg_model(image_640)
            output = interp(output2).cpu().data[0].numpy()
            del output1
            del output2

            # output_A = output.copy()
            # for idx, cl in enumerate(open_class):
            #     output_A = np.insert(output_A, cl, values=-100, axis=0)
            # output_A = output_A.transpose(1,2,0)
            # output_A = np.asarray(np.argmax(output_A, axis=2))

            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2))
            output1 = output.copy()
            for src, trg in enumerate(class_mapping):
                output[output1 == src] = trg

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
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100 * 19 / 16, 2)))
    return round(np.nanmean(mIoUs) * 100 * 19 / 16, 2)



def generate_pseudo_label16(seg_model, pred_dir=None, devkit_dir='/home/xiaoqiguo2/SimT_Plus/dataset/cityscapes_list', post=False):
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
    num_classes = np.int(info['closed classes']) + np.int(info['open classes'])
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

            # output1, output2 = seg_model(image)
            # output = interp(output2).cpu().data[0].numpy()
            # del output1
            # del output2

            output1, output2 = seg_model(image_640)
            output = interp(output2).cpu().data[0].numpy()
            del output1
            del output2

            ##### For AdaptsegNet, with 19 outputs 
            # output16 = np.copy(output)
            # for idx, cl in enumerate(open_class):
            #     output16 = np.delete(output16, cl-idx, 0)
            # output16 = output16.transpose(1,2,0)
            # output16 = np.asarray(np.argmax(output16, axis=2))
            # print(np.unique(output16))

            # for cl in open_class:
            #     output[cl,:,:] = np.zeros_like(output16) - 100.
            # output = output.transpose(1,2,0)
            # output = np.asarray(np.argmax(output, axis=2))

            ##### For BAPA-Net, with 16 outputs 
            output = output.transpose(1,2,0)
            output16 = np.asarray(np.argmax(output, axis=2))
            print(np.unique(output16))
            output = output16.copy()
            for src, trg in enumerate(class_mapping):
                output[output16 == src] = trg

            gt_dir ='/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes/train_label' # train_label, label
            gt_path = '%s/%s' % (gt_dir, name[0].split('leftImg8bit')[0]+'gtFine_labelIds.png')

            label = np.array(Image.open(gt_path))
            label = label_mapping(label, mapping)
            if len(label.flatten()) != len(output.flatten()):
                print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
                continue
            hist += fast_hist(label.flatten(), output.flatten(), num_classes)

            # output_col = colorize_mask(np.uint8(output))  ### 0-19
            # output = Image.fromarray(np.uint8(output16))  ### 0-16
            # name = name[0].split('/')[-1]
            # pred_dir = '/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes/Synthia/pseudo_bapa'
            # output.save('%s/%s' % (pred_dir, name))
            # output_col.save('%s/%s_color.png' % (pred_dir, name.split('.')[0]))

    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    return round(np.nanmean(mIoUs) * 100, 2)
