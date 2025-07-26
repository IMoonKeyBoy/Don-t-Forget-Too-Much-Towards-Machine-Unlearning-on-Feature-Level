import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from cprint import cprint
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.augmentations import get_transforms
from dataset.dataset import CelebASingleClassifiation
from functions import get_optimizer, get_scheduler, get_lr, get_single_classification_model
import sys
from copy import copy, deepcopy

import torch
import torch.nn as nn
import torch.nn.utils
from cprint import cprint
from torch import tensor

from Similar_Mask_Generate import SMGBlock
from functions import get_single_classification_model
from load_utils import load_state_dict_from_url
from newPad2d import newPad2d
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_pil_image

from load_utils import load_state_dict_from_url
from cub_voc import CUB_VOC
import os
from tqdm import tqdm, trange
import shutil
import numpy as np
import cv2 as cv
from Similar_Mask_Generate import SMGBlock
from newPad2d import newPad2d

F_MAP_SIZE = 196
CHANNEL_NUM = 256

_all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
          'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
          'wide_resnet50_2', 'wide_resnet101_2']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation)  # new padding


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.pad2d = newPad2d(1)  # new paddig

    def forward(self, x):
        identity = x
        out = self.pad2d(x)  # new padding
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pad2d(out)  # new padding
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.pad2d = newPad2d(1)  # new paddig

    def forward(self, x):
        identity = x
        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.pad2d(out)  # new padding
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=0,
                               bias=False)  # new padding
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)  # new padding
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.smg = SMGBlock(channel_size=CHANNEL_NUM, f_map_size=F_MAP_SIZE)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.pad2d_1 = newPad2d(1)  # new paddig
        self.pad2d_3 = newPad2d(3)  # new paddig

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, eval=False):
        # See note [TorchScript super()]
        x = self.pad2d_3(x)  # new padding
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pad2d_1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        encoded_image = deepcopy(x.detach())
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, encoded_image


def _resnet(arch, block, layers, num_class, pretrained, progress, **kwargs):
    model = ResNet(block, layers, num_class, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        pretrained_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def ResNet18(num_class, pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], num_class, pretrained, progress, **kwargs)


def ResNet34(num_class, pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], num_class, pretrained, progress, **kwargs)


def ResNet50(num_class, pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], num_class, pretrained, progress, **kwargs)


def ResNet101(num_class, pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], num_class, pretrained, progress, **kwargs)


def ResNet152(num_class, pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], num_class, pretrained, progress, **kwargs)


def get_cluster(self, matrix):
    cluser = []
    visited = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        tmp = []
        if (visited[i] == 0):
            for j in range(matrix.shape[1]):
                if (matrix[i][j] == 1):
                    tmp.append(j)
                    visited[j] = 1;
            cluser.append(tmp)
    return cluser


def channel_max_min_whole(f_map):
    T, C, H, W = f_map.shape
    max_v = np.max(f_map, axis=(0, 2, 3), keepdims=True)
    min_v = np.min(f_map, axis=(0, 2, 3), keepdims=True)
    return (f_map - min_v) / (max_v - min_v + 1e-6)


def train_one_epoch(epoch,
                    device,
                    model,
                    loader,
                    optimizer,
                    encoder):
    model.train()

    adv_preds = []
    encode_adv_preds = []
    adv_truth = []
    train_loss_list = list()

    pbar = tqdm(loader, total=loader.__len__(), position=0, leave=True)
    for sample in pbar:
        imgs, labels = sample['imgs'].float().to(device), sample['labels'].float().to(device)

        outputs_label, output_images = encoder(imgs)
        loss = np.load("../icCNN-main-new/icCNN/resnet/18_resnet_celeb_iccnn_59/loss_2500.npz")
        gt = loss['gt'][-1]  # show channel id of different groups
        cluster_label = get_cluster(gt)
        save_channel = []
        for i in range(len(cluster_label)):
            for j in range(len(cluster_label[i])):
                save_channel.append(cluster_label[i][j])
                break
        output_images = channel_max_min_whole(output_images.cpu().detach().numpy())
        new_training_images = tensor(np.zeros_like(imgs.cpu().detach().numpy()))
        for index_number in range(output_images.shape[0]):  # for each image
            fig = output_images[index_number][47]
            fig = cv.resize(fig, (112, 112)) * 255.0
            zeros_mask = cv.resize(fig, (224, 224))
            img = to_pil_image(imgs[index_number])
            mask_img = cv.resize(np.asarray(img), (224, 224))
            # Combine
            for i in range(zeros_mask.shape[0]):
                for j in range(zeros_mask.shape[1]):
                    if np.sum((zeros_mask[i][j] / 255.0) > 0.2):
                        mask_img[i][j] = 0
            new_training_images[index_number] = tensor(deepcopy(mask_img.transpose(2, 0, 1))).cuda()

        probs = model(imgs.cuda())
        encode_task_pred = model(new_training_images.cuda())

        loss = F.cross_entropy(encode_task_pred, labels.max(1, keepdim=False)[1], reduction='mean')
        train_loss = loss.item()
        train_loss_list.append(train_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = np.argmax(F.softmax(probs, dim=1).cpu().detach().numpy(), axis=1)
        adv_preds += probs.tolist()
        adv_acc = accuracy_score(adv_truth, adv_preds)

        encode_task_pred = np.argmax(F.softmax(encode_task_pred, dim=1).cpu().detach().numpy(), axis=1)
        encode_adv_preds += encode_task_pred.tolist()
        encode_adv_acc = accuracy_score(adv_truth, encode_adv_preds)

        adv_truth += labels.cpu().max(1, keepdim=False)[1].numpy().tolist()

        desc = f"Train Epoch : {epoch + 1}, Loss : {np.mean(train_loss_list):.2f}"
        desc = desc + ", Accuray" + f": {adv_acc:.2f}"
        desc = desc + ", encode_adv_acc" + f": {encode_adv_acc:.2f}"
        pbar.set_description(desc)

    adv_acc = accuracy_score(adv_truth, adv_preds)
    encode_adv_acc = accuracy_score(adv_truth, encode_adv_preds)

    return np.mean(train_loss_list), adv_acc, encode_adv_acc


def valid_one_epoch(epoch,
                    device,
                    model,
                    loader, encoder):
    adv_preds = []
    adv_truth = []
    encode_adv_preds = []

    train_loss_list = list()

    pbar = tqdm(loader, total=loader.__len__(), position=0, leave=True)
    for sample in pbar:
        imgs, labels = sample['imgs'].float().to(device), sample['labels'].float().to(device)

        outputs_label, output_images = encoder(imgs)
        loss = np.load("../icCNN-main-new/icCNN/resnet/18_resnet_celeb_iccnn_59/loss_2500.npz")
        gt = loss['gt'][-1]  # show channel id of different groups
        cluster_label = get_cluster(gt)
        save_channel = []
        for i in range(len(cluster_label)):
            for j in range(len(cluster_label[i])):
                save_channel.append(cluster_label[i][j])
                break
        output_images = channel_max_min_whole(output_images.cpu().detach().numpy())
        new_training_images = tensor(np.zeros_like(imgs.cpu().detach().numpy()))
        for index_number in range(output_images.shape[0]):  # for each image
            fig = output_images[index_number][47]
            fig = cv.resize(fig, (112, 112)) * 255.0
            zeros_mask = cv.resize(fig, (224, 224))
            img = to_pil_image(imgs[index_number])
            mask_img = cv.resize(np.asarray(img), (224, 224))
            # Combine
            for i in range(zeros_mask.shape[0]):
                for j in range(zeros_mask.shape[1]):
                    if np.sum((zeros_mask[i][j] / 255.0) > 0.2):
                        mask_img[i][j] = 0
            new_training_images[index_number] = tensor(deepcopy(mask_img.transpose(2, 0, 1))).cuda()

        probs = model(imgs.cuda())
        encode_task_pred = model(new_training_images.cuda())

        probs = np.argmax(F.softmax(probs, dim=1).cpu().detach().numpy(), axis=1)
        adv_preds += probs.tolist()
        adv_acc = accuracy_score(adv_truth, adv_preds)

        encode_task_pred = np.argmax(F.softmax(encode_task_pred, dim=1).cpu().detach().numpy(), axis=1)
        encode_adv_preds += encode_task_pred.tolist()
        encode_adv_acc = accuracy_score(adv_truth, encode_adv_preds)

        adv_truth += labels.cpu().max(1, keepdim=False)[1].numpy().tolist()

        desc = f"Train Epoch : {epoch + 1}, Loss : {np.mean(train_loss_list):.2f}"
        desc = desc + ", Accuray" + f": {adv_acc:.2f}"
        desc = desc + ", encode_adv_acc" + f": {encode_adv_acc:.2f}"
        pbar.set_description(desc)

    adv_acc = accuracy_score(adv_truth, adv_preds)
    encode_adv_acc = accuracy_score(adv_truth, encode_adv_preds)

    return adv_acc, encode_adv_acc


def training(args):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.empty_cache()

    args.save_path = os.path.join(args.save_path, args.run_name)
    if os.path.exists(args.save_path):
        cprint.err("Folder is exiting!")
    else:
        os.makedirs(args.save_path)

    # Prepare for data
    train_transforms = get_transforms(args.img_size, mode='train')
    valid_transforms = get_transforms(args.img_size, mode='valid')

    train_dataset = CelebASingleClassifiation(args.data_path,
                                              args.task_labels,
                                              transforms=train_transforms,
                                              is_split_notask_and_task_dataset=False,
                                              is_training=True)
    valid_dataset = CelebASingleClassifiation(args.data_path,
                                              args.task_labels,
                                              transforms=valid_transforms,
                                              is_split_notask_and_task_dataset=False,
                                              is_training=False)

    train_loader = DataLoader(train_dataset,
                              args.batch_size,
                              drop_last=True,
                              num_workers=6,
                              pin_memory=use_cuda)
    valid_loader = DataLoader(valid_dataset,
                              args.batch_size,
                              drop_last=True,
                              num_workers=6,
                              pin_memory=use_cuda)

    model = get_single_classification_model(args.model)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model.to(device)

    encoder_network = ResNet18(num_class=80, pretrained=False).cuda()
    pretrained_dict = torch.load(args.pretrained_network_path)
    encoder_network.load_state_dict(pretrained_dict)

    if args.wandb:
        wandb.init(project='Feature unlearning',
                   group=args.group_name,
                   name=args.run_name,
                   config=args)
        wandb.watch(model)

    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args.scheduler, optimizer)

    if args.is_training_process:
        for epo in range(args.epoch):

            with torch.set_grad_enabled(True):
                train_loss, train_accuracies, av_train_accuracies = train_one_epoch(epoch=epo,
                                                                                    device=device,
                                                                                    model=model,
                                                                                    loader=train_loader,
                                                                                    optimizer=optimizer,
                                                                                    encoder=encoder_network)
            logs = {"Train train_accuracies": train_accuracies}
            logs["Train av_train_accuracies"] = av_train_accuracies

            with torch.no_grad():
                valid_accuracies, ad_train_accuracies = valid_one_epoch(epoch=epo,
                                                                        device=device,
                                                                        model=model,
                                                                        loader=valid_loader,
                                                                        encoder=encoder_network)

            logs["Valid valid_accuracies"] = valid_accuracies
            logs["Valid ad_train_accuracies"] = ad_train_accuracies

            if args.wandb:
                wandb.log(logs)

            if scheduler:
                scheduler.step()

            if epo % 5 == 0:
                print(args.save_path)
                torch.save(model, os.path.join(args.save_path, args.run_name + "_" + str(epo) + ".pth"))

        torch.save(model, os.path.join(args.save_path, args.run_name + "_final.pth"))
        if args.wandb:
            wandb.join()


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../datasets/CelebaK/', help='Root path of data')
    parser.add_argument('--save_path', type=str, default='./checkpoints/original_norm/', help='Dir path to save model weights')

    parser.add_argument('--model', type=str, default='customresnet', choices=('customresnet', 'customefficientnet', 'effnetv2s'))
    parser.add_argument('--epoch', type=int, default=25, help='Total Number of epochs')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-06, help='Learning Rate')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=('adam', 'adamw'))
    parser.add_argument('--weight_decay', type=float, default=1e-06)
    parser.add_argument('--img_size', type=int, default=224, help='Size of input image')
    parser.add_argument('--scheduler', type=str, default='cosinewarmup', choices=('none', 'cosinewarmup'))

    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--group_name', type=str, default='Single Classification Training Process')
    parser.add_argument('--wandb', default=False, action='store_true', help='Use wandb for logging')

    parser.add_argument('--multi', default=False, action='store_true', help='Use wandb for logging')
    parser.add_argument('--is_training_process', default=True, action='store_true', help='Use wandb for logging')

    # Male, Young, Pointy_Nose, Bald, Smiling, Arched_Eyebrows,Narrow_Eyes,Mustache,Wearing_Hat,Wearing_Earrings,Big_Lips
    parser.add_argument('--task_labels', type=list, default=['Young'], help='Attributes')


    args = parser.parse_args()
    args.run_name = '_'.join(args.task_labels)

    args.pretrained_network_path = '../icCNN-main-new/icCNN/resnet/18_resnet_celeb_iccnn_59/model_2500.pth'


    for k, v in sorted(vars(args).items()):
        cprint.info(k, '=', v)

    training(args)
