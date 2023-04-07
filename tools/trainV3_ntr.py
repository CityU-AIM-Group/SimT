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

from model.deeplab import Deeplab, sig_NTM, sig_W, FullyConnectGCLayer
# from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d, EntropyLoss
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesPseudo

import time
import datetime
import itertools
import pandas as pd
import _init_paths
from evaluate_cityscapes import evaluate_simt_unknown, evaluate_simt_3unknown, evaluate_warmup
import matplotlib.pyplot as plt
plt.switch_backend('agg')


IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 2
NUM_WORKERS = 4
DATA_DIRECTORY = '/home/xiaoqiguo2/scratch/UDA_Natural/GTA5'
DATA_LIST_PATH = '../dataset/gta5_list/train.txt'
Word_embedding_path = '../prior/WordEmb19.npy'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,640'
DATA_DIRECTORY_TARGET = '/home/xiaoqiguo2/scratch/UDA_Natural/Cityscapes'
DATA_LIST_PATH_TARGET = '../dataset/cityscapes_list/pseudo_bapa.lst'
INPUT_SIZE_TARGET = '1280,640'
LEARNING_RATE = 2.5e-4
LEARNING_RATE_T = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
OPEN_CLASSES = 15
NUM_STEPS = 250000
NUM_STEPS_STOP = 10000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = '../snapshots/resnet_pretrain.pth'
RESTORE_FROM_NOISE = '../snapshots/resnet_pretrain.pth'
SAVE_PRED_EVERY = 500
SNAPSHOT_DIR = '../snapshots/SimT/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log/'
Threshold_conf = 0.8
temperature = 4.
lambda_clean = 0.1
lambda_relation = 1.0
lambda_WE = 0.1
lambda_Convex = 0.1
lambda_Volume = 0.001
lambda_Anchor = 1.0

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
    parser.add_argument("--Threshold-conf", type=float, default=Threshold_conf,
                        help="Threshold_conf")
    parser.add_argument("--temperature", type=float, default=temperature,
                        help="temperature")
    parser.add_argument("--lambda-clean", type=float, default=lambda_clean,
                        help="lambda_clean")
    parser.add_argument("--lambda-relation", type=float, default=lambda_relation,
                        help="lambda_relation")
    parser.add_argument("--lambda-WE", type=float, default=lambda_WE,
                        help="lambda_WE")
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
    parser.add_argument("--restore-from-noise", type=str, default=RESTORE_FROM_NOISE,
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
# print('Leanring_rate_T: ', args.learning_rate_T)
print('Closed-set class: ', args.num_classes)
print('Open-set class: ', args.open_classes)
# print('Threshold_conf: ', args.Threshold_conf)
print('temperature: ', args.temperature)
print('lambda_clean: ', args.lambda_clean)
print('lambda_relation: ', args.lambda_relation)
# print('lambda_WE: ', args.lambda_WE)
# print('lambda_Convex: ', args.lambda_Convex)
# print('lambda_Volume: ', args.lambda_Volume)
# print('lambda_Anchor: ', args.lambda_Anchor)
print('restore_from: ', args.restore_from)
print('restore_from_noise: ', args.restore_from_noise)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return f_x

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_T(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_T, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr

def Relation_loss(prediction, pseudo, simi_relation, num_classes=args.num_classes, open_classes=args.open_classes, thres=args.Threshold_conf):
    kl_loss = nn.KLDivLoss(reduce=False)
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    ones = torch.ones_like(pseudo)

    clean_closed = torch.where(pseudo == 255, (num_classes) * ones, pseudo).long()
    clean_closed_onehot = torch.eye(num_classes + 1)[clean_closed].permute(0, 3, 1, 2).float().cuda()[:, :num_classes, :, :]
    bs, c, h, w = clean_closed_onehot.shape
    relation_label = clean_closed_onehot.permute(0, 2, 3, 1).view(-1, c).mm(simi_relation)
    relation_label = relation_label.view(bs, h, w, c+open_classes).permute(0, 3, 1, 2) ### bs * 19 * h * w
    clean_closed_onehot = torch.eye(num_classes + open_classes + 1)[clean_closed].permute(0, 3, 1, 2).float().cuda()[:, :num_classes + open_classes, :, :]
    predict = torch.where(clean_closed_onehot > 0, -100.*torch.ones_like(prediction), prediction)

    loss_relation = kl_loss((relation_label+10e-10).log(), torch.softmax(predict, dim=1)+10e-10)
    # loss_relation = kl_loss(( torch.softmax(predict, dim=1) + 10e-10 ).log(), relation_label+10e-10)
    weight = torch.ones_like(clean_closed_onehot) - clean_closed_onehot
    mask = torch.where(pseudo == 255, torch.zeros_like(pseudo), ones).float()
    loss_relation = torch.sum(torch.sum(loss_relation.mul(weight), dim=1).mul(mask)) / (torch.sum(mask)+10e-10)
    return loss_relation

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
    model = Deeplab(num_classes=args.num_classes, open_classes=args.open_classes, openset=True).cuda()
    net_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict)}
    # pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in net_dict) and (v.shape==net_dict[k[6:]].shape)}
    net_dict.update(pretrained_dict)
    model.load_state_dict(net_dict)
    model.train()

    # Create fixed network
    pretrained_dict = torch.load(args.restore_from_noise)
    fixed_model = Deeplab(num_classes=args.num_classes).cuda()
    net_dict = fixed_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict)}
    net_dict.update(pretrained_dict)
    fixed_model.load_state_dict(net_dict)
    fixed_model.eval()
    for param in fixed_model.parameters():
        param.requires_grad = False

    cudnn.benchmark = True

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    # Create dataloader
    targetloader = data.DataLoader(cityscapesPseudo(args.data_dir_target, args.data_list_target,
                    max_iters=args.num_steps * args.batch_size,
                    crop_size=input_size_target,
                    scale=False, mirror=args.random_mirror, mean=IMG_MEAN),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True)

    targetloader_iter = enumerate(targetloader)

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    #### Calculate similar and non-similar relations with word embedding ####
    Word_embedding = torch.from_numpy(np.load(Word_embedding_path)).float()
    Word_embedding = torch.unsqueeze(Word_embedding, 0).permute(0,2,1).cuda() ## b, c, n
    WE = Word_embedding.clone().detach().permute(0,2,1)
    we_relation = torch.cdist(WE, WE.permute(1,0,2), p=2.0)[:,:,0]
    we_relation = we_relation.max() - we_relation
    weight = torch.ones(args.num_classes + args.open_classes) - torch.eye(args.num_classes + args.open_classes)
    we_relation = we_relation.mul(weight.cuda()) - 10 * torch.eye(args.num_classes + args.open_classes).cuda()
    we_relation = torch.softmax(we_relation * 2, dim=1)[:args.num_classes,:]

    # Create NTM
    restore_file = args.restore_from.replace(SNAPSHOT_DIR, "../SimT_numpy/") 
    restore_file = restore_file.replace('.pth', ".npy") 
    T = torch.from_numpy(np.load(restore_file)).cuda()

    temp = args.temperature
    T_inv = np.linalg.pinv(T.detach().cpu().numpy())
    T_inv = torch.from_numpy(softmax(T_inv * temp)).float().cuda()

    # Create loss function
    Info_loss = EntropyLoss().cuda()
    kl_loss = nn.KLDivLoss(reduce=False)
    kl_loss_reduce = nn.KLDivLoss(reduce=True)
    loss_mse = torch.nn.MSELoss(reduction='sum').cuda()
    Tseg_loss = CrossEntropy2d(is_softmax=False).cuda()
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
    for i_iter in range(args.num_steps):
        model.train()
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        for sub_i in range(args.iter_size):
            _, batch = targetloader_iter.__next__()
            image_target, label_target, _, name = batch
            image_target = image_target.cuda()
            label_target = label_target.long().cuda()

            with torch.no_grad():
                _, output = fixed_model(image_target)
                noise_class_posterior_sup = interp_target(torch.softmax(output.clone().detach(), dim=1))
                noise_class_posterior_flat = noise_class_posterior_sup.permute(0,2,3,1).view(-1, args.num_classes)
                del output

            ###############################
            ##### Train target images #####
            ###############################
            _, pred = model(image_target)
            pred = interp_target(pred)

           ############################# Optimize segmentation net ############################

            ######## Relation loss ########
            WE_relation_loss = Relation_loss(pred, label_target, we_relation)

            ######## Noise class posterior constraint ########
            clean_pred = torch.softmax(interp_target(pred), dim=1)
            noise_pred = clean_pred.permute(0, 2, 3, 1).contiguous().view(-1, args.num_classes + args.open_classes)
            noise_pred = torch.mm(noise_pred, T).view(args.batch_size, h, w, args.num_classes).permute(0, 3, 1, 2)
            loss_y = Tseg_loss(noise_pred, label_target)
            loss_y += kl_loss_reduce((noise_pred + 10e-10).log(), noise_class_posterior_sup + 10e-10) #+ 0.01 * en_loss(pred)

            ######## Clean class posterior constraint ########
            bs, c, h, w = noise_class_posterior_sup.shape  
            clean_class_posterior_sup = noise_class_posterior_sup.permute(0, 2, 3, 1).contiguous().view(-1, args.num_classes).mm(T_inv)
            clean_class_posterior_sup = clean_class_posterior_sup.view(bs, h, w, args.num_classes+args.open_classes).permute(0, 3, 1, 2)
            loss_p = kl_loss_reduce((clean_pred + 10e-10).log(), clean_class_posterior_sup + 10e-10) 

            # joint loss function
            loss = loss_y + args.lambda_clean * loss_p + args.lambda_relation * WE_relation_loss 

            # proper normalization
            loss = loss / args.iter_size
            loss.backward()

        optimizer.step()
        if (i_iter) % 100 == 0:
            print(
            'iter = {0:8d}/{1:8d}, loss_y = {2:.3f} loss_p = {3:.3f} relation = {4:.3f}'.format(
                i_iter, args.num_steps, loss_y, loss_p, WE_relation_loss))

        if i_iter >= args.num_steps_stop - 1:
            # print('save model ...')
            # torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            now = datetime.datetime.now()
            print(now.strftime("%Y-%m-%d %H:%M:%S"), '  Begin evaluation on iter {0:8d}/{1:8d}  '.format(i_iter, args.num_steps))
            mIoU = evaluate_simt_3unknown(model, with_prior=True)
            print('Finish Evaluation: '+time.asctime(time.localtime(time.time())))
            if mIoU > best_mIoU:
                old_file = osp.join(args.snapshot_dir, 'Synthia_iter' + str(best_iter) + '_HOS' + str(best_mIoU) + '.pth')
                if os.path.exists(old_file) is True:
                    os.remove(old_file) 
                print('Saving model with HOS: ', mIoU)
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'Synthia_iter' + str(i_iter) + '_HOS' + str(mIoU) + '.pth'))
                best_mIoU = mIoU
                best_iter = i_iter


if __name__ == '__main__':
    main()
