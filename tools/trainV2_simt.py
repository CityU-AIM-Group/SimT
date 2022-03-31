import _init_paths
import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random

from model.deeplab_multi import DeeplabMulti, sig_NTM, sig_W
# from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d, EntropyLoss
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet, cityscapesPseudo

import time
import datetime
import itertools
import pandas as pd
import _init_paths
from evaluate_cityscapes import evaluate_simt
import matplotlib.pyplot as plt
plt.switch_backend('agg')


IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '/home/xiaoqiguo2/scratch/UDA_Natural/GTA5'
DATA_LIST_PATH = '../dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1024,512'
DATA_DIRECTORY_TARGET = '/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes'
DATA_LIST_PATH_TARGET = '../dataset/cityscapes_list/pseudo_bapa.lst'
INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 2.5e-4
LEARNING_RATE_T = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
OPEN_CLASSES = 15
NUM_STEPS = 250000
NUM_STEPS_STOP = 40000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = '../snapshots/resnet_pretrain.pth'
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = '../snapshots/SimT/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log/'
Threshold_high = 0.8
Threshold_low = 0.2
lambda_Place = 0.1
lambda_Convex = 0.5
lambda_Volume = 0.1
lambda_Anchor = 0.5

LAMBDA_SEG = 0.1
TARGET = 'cityscapes'
SET = 'train'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-T", type=float, default=LEARNING_RATE_T,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--Threshold-high", type=float, default=Threshold_high,
                        help="Threshold_high")
    parser.add_argument("--Threshold-low", type=float, default=Threshold_low,
                        help="Threshold_low")
    parser.add_argument("--lambda-Place", type=float, default=lambda_Place,
                        help="lambda_Place")
    parser.add_argument("--lambda-Convex", type=float, default=lambda_Convex,
                        help="lambda_Convex")
    parser.add_argument("--lambda-Volume", type=float, default=lambda_Volume,
                        help="lambda_Volume")
    parser.add_argument("--lambda-Anchor", type=float, default=lambda_Anchor,
                        help="lambda_Anchor")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--open-classes", type=int, default=OPEN_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    return parser.parse_args()


args = get_arguments()
# if not os.path.exists(args.log_dir):
#     os.makedirs(args.log_dir)
print('Leanring_rate: ', args.learning_rate)
print('Leanring_rate_T: ', args.learning_rate_T)
print('Open-set class: ', args.open_classes)
print('Threshold_high: ', args.Threshold_high)
print('Threshold_low: ', args.Threshold_low)
print('lambda_Place: ', args.lambda_Place)
print('lambda_Convex: ', args.lambda_Convex)
print('lambda_Volume: ', args.lambda_Volume)
print('lambda_Anchor: ', args.lambda_Anchor)
print('restore_from: ', args.restore_from)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_T(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_T, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr

def plot_NTM(trans_mat, normalize=True, title='NTM1', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(trans_mat, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    thresh = trans_mat.max() / 2.
    for i, j in itertools.product(range(trans_mat.shape[0]), range(trans_mat.shape[1])):
        num = '{:.2f}'.format(trans_mat[i, j]) if normalize else int(trans_mat[i, j])
        plt.text(j, i, num,
                 fontsize=2, 
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if np.float(num) > thresh else "black")
    plt.savefig('../NTM_vis/'+title+'.png', transparent=True, dpi=600)

def Placeholder_loss(pred, num_classes, open_classes, thres=None):
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
    #### del maximum elements in prediction####
    pseudo = torch.argmax(pred, dim=1).long()
    pseudo_onehot = torch.eye(num_classes + open_classes)[pseudo].permute(0, 3, 1, 2).float().cuda()
    zeros = torch.zeros_like(pseudo_onehot)
    ones = torch.zeros_like(pseudo_onehot)
    predict = torch.where(pseudo_onehot > zeros, -1000. * ones, pred)

    #### del pixels with armgmax < num_classes ####
    ones = torch.ones_like(pseudo)
    pseudo1 = torch.where(pseudo < num_classes * ones, pseudo, 255 * ones)
    if thres is not None:
        pred_max = torch.max(torch.softmax(pred.clone().detach(), dim=1), 1)[0]
        pseudo1 = torch.where(pred_max > thres, pseudo1, 255 * ones)
    loss_known = seg_loss(pred, pseudo1)

    #### find out the maximum logit within open set classes as the label ####
    predict_open = torch.zeros_like(predict)
    predict_open[:,args.num_classes:,:,:] = predict[:,args.num_classes:,:,:].clone().detach()
    Placeholder_y = torch.argmax(predict_open, dim=1)
    Placeholder_y = torch.where(pseudo1 == 255 * ones, 255 * ones, Placeholder_y)

    # yy = torch.where(pseudo1 == 255 * ones, (num_classes + open_classes) * ones, Placeholder_y)
    # Placeholder_y_onehot = torch.eye(num_classes + open_classes + 1)[yy].permute(0, 3, 1, 2).float().cuda()[:,:(num_classes + open_classes),:,:]
    # predict[:,args.num_classes:,:,:] = torch.where(Placeholder_y_onehot > zeros, predict, -1000. * ones)[:,args.num_classes:,:,:]

    loss_unknown = seg_loss(predict, Placeholder_y)
    return loss_known + args.lambda_Place * loss_unknown

def main():
    """Create the model and start the training."""
    print('Start: '+time.asctime(time.localtime(time.time())))
    best_iter = 0
    best_mIoU = 0
    mIoU = 0

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True
    gpu = args.gpu

    pretrained_dict = torch.load(args.restore_from)
    # Create network
    model = DeeplabMulti(num_classes=args.num_classes, open_classes=args.open_classes, openset=True).cuda()
    net_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict)}
    # pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in net_dict) and (v.shape==net_dict[k[6:]].shape)}
    net_dict.update(pretrained_dict)
    model.load_state_dict(net_dict)
    model.train()

    # Create fixed network
    pretrained_dict = torch.load(args.restore_from)
    fixed_model = DeeplabMulti(num_classes=args.num_classes).cuda()
    net_dict = fixed_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict)}
    net_dict.update(pretrained_dict)
    fixed_model.load_state_dict(net_dict)
    fixed_model.eval()
    for param in fixed_model.parameters():
        param.requires_grad = False

    # Create NTM
    NTM1 = sig_NTM(args.num_classes, args.open_classes)
    optimizer_t1 = optim.Adam(NTM1.parameters(), lr=args.learning_rate_T, weight_decay=0)

    NTM2 = sig_NTM(args.num_classes, args.open_classes)
    optimizer_t2 = optim.Adam(NTM2.parameters(), lr=args.learning_rate_T, weight_decay=0)

    NTM_W1 = sig_W(args.num_classes, args.open_classes)
    optimizer_w1 = optim.Adam(NTM_W1.parameters(), lr=args.learning_rate_T, weight_decay=0)

    NTM_W2 = sig_W(args.num_classes, args.open_classes)
    optimizer_w2 = optim.Adam(NTM_W2.parameters(), lr=args.learning_rate_T, weight_decay=0)

    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    targetloader = data.DataLoader(cityscapesPseudo(args.data_dir_target, args.data_list_target,
                    max_iters=args.num_steps * args.batch_size,
                    crop_size=input_size_target,
                    scale=False, mirror=args.random_mirror, mean=IMG_MEAN),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True)

    targetloader_iter = enumerate(targetloader)

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
    Tseg_loss = CrossEntropy2d(is_softmax=False).cuda()
    loss_mse = torch.nn.MSELoss(reduction='sum').cuda()
    Info_loss = EntropyLoss().cuda()
    for i_iter in range(args.num_steps):
        model.train()
        loss_seg_p1 = 0
        loss_seg_p2 = 0
        loss_seg_y1 = 0
        loss_seg_y2 = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_t1.zero_grad()
        optimizer_t2.zero_grad()
        optimizer_w1.zero_grad()
        optimizer_w2.zero_grad()
        adjust_learning_rate_T(optimizer_t1, i_iter)
        adjust_learning_rate_T(optimizer_t2, i_iter)
        adjust_learning_rate_T(optimizer_w1, i_iter)
        adjust_learning_rate_T(optimizer_w2, i_iter)

        zeros =  torch.zeros(args.num_classes+args.open_classes, args.num_classes).cuda()
        for iter in range(10):
            ## optimize weight ###
            T1 = NTM1()
            T2 = NTM2()
            W1 = NTM_W1()
            W2 = NTM_W2()
            optimizer_w1.zero_grad()
            optimizer_w2.zero_grad()
                
            NTM_loss = (loss_mse(W1.mm(T1), zeros) + loss_mse(W2.mm(T2), zeros))
            NTM_loss.backward(retain_graph=True)
            optimizer_w1.step()
            optimizer_w2.step()

        for sub_i in range(args.iter_size):
            T1 = NTM1()
            T2 = NTM2()

            _, batch = targetloader_iter.__next__()
            image_target, label_target, _, name = batch
            image_target = image_target.cuda()
            label_target = label_target.long().cuda()

            ##### Generate pseudo label #####
            with torch.no_grad():
                fixed_model.load_state_dict(net_dict)
                output1, output2 = fixed_model(image_target)
                labelC = interp_target(torch.softmax(output2.clone(), dim=1))
                labelC_max = torch.max(labelC, 1)
                labelC_argmax = torch.argmax(labelC, dim=1).float()
                labelC_flat = labelC.permute(0,2,3,1).view(-1, args.num_classes)
                thres = args.Threshold_high
                labelC = torch.where(labelC_max[0] > thres, labelC_argmax, 255. * torch.ones_like(labelC_argmax))
                thres = args.Threshold_low
                labelC = torch.where(labelC_max[0] < thres, args.num_classes * torch.ones_like(labelC_argmax), labelC)
                Conf_label_target = torch.from_numpy(labelC.detach().clone().cpu().numpy()).long().cuda()  
                del labelC           
                del output1
                del output2

            ###############################
            ##### Train target images #####
            ###############################
            pred1, pred2 = model(image_target)
            pred1 = interp_target(pred1)
            pred2 = interp_target(pred2)

            ######## Anchor loss ########
            pseudo_flat1 = pred1.clone().permute(0,2,3,1).view(-1, args.num_classes+args.open_classes).detach()
            Anchor_index = torch.argmax(pseudo_flat1, dim=0)
            Exist_label = torch.unique(torch.argmax(pseudo_flat1, dim=1))
            Anchor1 = labelC_flat[Anchor_index]
            NTM_Anchor_loss =  loss_mse(T1[Exist_label], Anchor1[Exist_label])
            pseudo_flat2 = pred2.clone().permute(0,2,3,1).view(-1, args.num_classes+args.open_classes).detach()
            Anchor_index = torch.argmax(pseudo_flat2, dim=0)
            Exist_label = torch.unique(torch.argmax(pseudo_flat2, dim=1))
            Anchor2 = labelC_flat[Anchor_index]
            NTM_Anchor_loss += loss_mse(T2[Exist_label], Anchor2[Exist_label])

            ######## Class posterior constraint ########
            pseudo = torch.argmax(pred2.clone(), dim=1).detach()
            ones = torch.ones_like(Conf_label_target)
            zeros = torch.zeros_like(Conf_label_target)
            mask = torch.where(Conf_label_target == args.num_classes * ones, ones, zeros)
            pseudo1 = mask * pseudo
            pseudo1 = torch.where(pseudo1 >= args.num_classes * ones, pseudo1, 255 * ones)
            Conf_label_target = torch.where(Conf_label_target == args.num_classes * ones, pseudo1, Conf_label_target)
            loss_p1 = seg_loss(pred1, Conf_label_target)
            loss_p2 = seg_loss(pred2, Conf_label_target)

            ######## Placeholder loss ########
            Place_loss = args.lambda_seg * Placeholder_loss(pred1, args.num_classes, args.open_classes, thres=args.Threshold_high)
            Place_loss += Placeholder_loss(pred2, args.num_classes, args.open_classes, thres=args.Threshold_high)

            ######## Noise class posterior constraint ########
            pred1 = torch.softmax(interp_target(pred1), dim=1).permute(0, 2, 3, 1).contiguous().view(-1, args.num_classes + args.open_classes)
            pred1 = torch.mm(pred1, T1).view(args.batch_size, h, w, args.num_classes).permute(0, 3, 1, 2)

            pred2 = torch.softmax(interp_target(pred2), dim=1).permute(0, 2, 3, 1).contiguous().view(-1, args.num_classes + args.open_classes)
            pred2 = torch.mm(pred2, T2).view(args.batch_size, h, w, args.num_classes).permute(0, 3, 1, 2)

            loss_y1 = Tseg_loss(pred1, label_target)
            loss_y2 = Tseg_loss(pred2, label_target)

            ## optimze NTM ###
            W1 = NTM_W1()
            W2 = NTM_W2()
            zeros =  torch.zeros(args.num_classes+args.open_classes, args.num_classes).cuda()
            NTM_Convex_loss = 0. - (loss_mse(W1.mm(T1), zeros) + loss_mse(W2.mm(T2), zeros))

            NTM_Volume_loss =  torch.log(torch.sqrt(torch.abs(torch.linalg.det( T1.transpose(1,0).mm(T1) ))))
            NTM_Volume_loss += torch.log(torch.sqrt(torch.abs(torch.linalg.det( T2.transpose(1,0).mm(T2) ))))

            if torch.isinf(NTM_Volume_loss) or torch.isnan(NTM_Volume_loss):
                NTM_Volume_loss = 0.

            loss_target = loss_p2 + loss_y2 + args.lambda_seg * loss_p1 + args.lambda_seg * loss_y1
            loss = Place_loss + loss_target + args.lambda_Convex * NTM_Convex_loss + args.lambda_Volume * NTM_Volume_loss + args.lambda_Anchor * NTM_Anchor_loss

            # proper normalization
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_p1 += loss_p1.data.cpu().numpy() / args.iter_size
            loss_seg_p2 += loss_p2.data.cpu().numpy() / args.iter_size
            loss_seg_y1 += loss_y1.data.cpu().numpy() / args.iter_size
            loss_seg_y2 += loss_y2.data.cpu().numpy() / args.iter_size

        optimizer.step()
        optimizer_t1.step()
        optimizer_t2.step()

        if (i_iter) % 100 == 0:
            print(
            'iter = {0:8d}/{1:8d}, loss_seg_p = {2:.3f} loss_seg_y = {3:.3f} Convex = {4:.3f} Volume = {5:.3f} Anchor = {6:.3f} Place_loss = {7:.3f}'.format(
                i_iter, args.num_steps, loss_seg_p1 + loss_seg_p2, loss_seg_y1 + loss_seg_y2, NTM_Convex_loss, NTM_Volume_loss, NTM_Anchor_loss, Place_loss))

        # if (i_iter) % 5000 == 0:
        #     plot_NTM(NTM1().detach().cpu().numpy(), normalize=True, title='NTM1_'+str(i_iter), cmap=plt.cm.Blues)
        #     plot_NTM(NTM2().detach().cpu().numpy(), normalize=True, title='NTM2_'+str(i_iter), cmap=plt.cm.Blues)

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            now = datetime.datetime.now()
            print(now.strftime("%Y-%m-%d %H:%M:%S"), '  Begin evaluation on iter {0:8d}/{1:8d}  '.format(i_iter, args.num_steps))
            mIoU = evaluate_simt(model)
            print('Finish Evaluation: '+time.asctime(time.localtime(time.time())))
            if mIoU > best_mIoU:
                old_file = osp.join(args.snapshot_dir, 'GTA5_iter' + str(best_iter) + '_mIoU' + str(best_mIoU) + '.pth')
                if os.path.exists(old_file) is True:
                    os.remove(old_file) 
                print('Saving model with mIoU: ', mIoU)
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_iter' + str(i_iter) + '_mIoU' + str(mIoU) + '.pth'))
                best_mIoU = mIoU
                best_iter = i_iter


if __name__ == '__main__':
    main()
