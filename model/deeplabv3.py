import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import numpy as np

class ResNet_50(nn.Module): 
    def __init__(self, in_channels = 3, conv1_out = 64):
        super(ResNet_50,self).__init__()
        self.resnet_50 = models.resnet50(pretrained = True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,x):
        x = self.relu(self.resnet_50.bn1(self.resnet_50.conv1(x)))
        x = self.resnet_50.maxpool(x)
        x = self.resnet_50.layer1(x)
        x = self.resnet_50.layer2(x)
        x = self.resnet_50.layer3(x)
        return x

class ASSP(nn.Module):
    def __init__(self,in_channels,out_channels = 256):
        super(ASSP,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 1,
                          padding = 0,
                          dilation=1,
                          bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride=1,
                          padding = 6,
                          dilation = 6,
                          bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride=1,
                          padding = 12,
                          dilation = 12,
                          bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride=1,
                          padding = 18,
                          dilation = 18,
                          bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        self.conv5 = nn.Conv2d(in_channels = in_channels, 
                          out_channels = out_channels,
                          kernel_size = 1,
                          stride=1,
                          padding = 0,
                          dilation=1,
                          bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        
        self.convf = nn.Conv2d(in_channels = out_channels * 5, 
                          out_channels = out_channels,
                          kernel_size = 1,
                          stride=1,
                          padding = 0,
                          dilation=1,
                          bias=False)
        self.bnf = nn.BatchNorm2d(out_channels)
        self.adapool = nn.AdaptiveAvgPool2d(1)  
        
    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        
        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)
        
        # x5 = self.adapool(x)
        x5 = self.conv5(x)
        x5 = self.bn5(x5)
        x5 = self.relu(x5)
        x5 = F.interpolate(x5, size = tuple(x4.shape[-2:]), mode='bilinear')
        
        x = torch.cat((x1,x2,x3,x4,x5), dim = 1) #channels first
        x = self.convf(x)
        x = self.bnf(x)
        x = self.relu(x)
        return x


class DeepLabv3(nn.Module):
    def __init__(self, nc, openc=0, openset=False):
        super(DeepLabv3, self).__init__()
        
        self.nc = nc
        self.openc = openc
        self.openset = openset
        
        self.resnet = ResNet_50()
        
        self.assp = ASSP(in_channels = 1024)
        
        self.conv = nn.Conv2d(in_channels = 256, out_channels = self.nc,
                          kernel_size = 1, stride=1, padding=0)
        if openset:
            self.conv_1 = nn.Conv2d(in_channels = 256, out_channels = self.openc,
                          kernel_size = 1, stride=1, padding=0)
                          
    def forward(self,x):
        _, _, h, w = x.shape
        x = self.resnet(x)
        x = self.assp(x)
        x1 = self.conv(x)
        if self.openset:
            x1_1 = self.conv_1(x)
            x1 = torch.cat([x1, x1_1], dim=1)
        x1 = F.interpolate(x1, size=(h, w), mode='bilinear') #scale_factor = 16, mode='bilinear')
        return x1

    def get_1x_lr_params_NOscale(self):
        b = []
        b.append(self.resnet)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.named_parameters():
                    jj += 1
                    if 'resnet_50.layer3' in k[0] or 'resnet_50.layer4' in k[0] or 'resnet_50.fc' in k[0]:
                        # if k.requires_grad:
                        yield k[1]

    def get_10x_lr_params(self):
        b = []
        b.append(self.assp.parameters())
        b.append(self.conv.parameters())
        if self.openset:
            b.append(self.conv_1.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]

class sig_NTM(nn.Module):
    def __init__(self, num_classes, open_classes=0, init = None):
        super(sig_NTM, self).__init__()

        T = torch.ones(num_classes+open_classes, num_classes)
        self.register_parameter(name='NTM', param=nn.parameter.Parameter(torch.FloatTensor(T)))
        self.NTM
        
        nn.init.kaiming_normal_(self.NTM, mode='fan_out', nonlinearity='relu')

        self.Identity_prior = torch.cat([torch.eye(num_classes, num_classes), torch.zeros(open_classes, num_classes)], 0)
        Class_dist = np.load('../ClassDist/ClassDist_source.npy')
        # Class_dist = Class_dist / Class_dist.max()
        self.Class_dist = torch.FloatTensor(np.tile(Class_dist, (num_classes + open_classes, 1)))

    def forward(self):
        T = torch.sigmoid(self.NTM).cuda()
        T = T.mul(self.Class_dist.cuda().detach()) + self.Identity_prior.cuda().detach()
        T = F.normalize(T, p=1, dim=1)
        return T

class sig_W(nn.Module):
    def __init__(self, num_classes, open_classes=0):
        super(sig_W, self).__init__()
        
        self.classes = num_classes+open_classes
        init = 1./(self.classes-1.)

        self.register_parameter(name='weight', param=nn.parameter.Parameter(init*torch.ones(self.classes, self.classes)))

        self.weight

        self.identity = torch.zeros(self.classes, self.classes) - torch.eye(self.classes)

    def forward(self):
        ind = np.diag_indices(self.classes)
        with torch.no_grad():
            self.weight[ind[0], ind[1]] = -10000. * torch.ones(self.classes).detach()

        w = torch.softmax(self.weight, dim = 1).cuda()

        weight = self.identity.detach().cuda() + w
        return weight