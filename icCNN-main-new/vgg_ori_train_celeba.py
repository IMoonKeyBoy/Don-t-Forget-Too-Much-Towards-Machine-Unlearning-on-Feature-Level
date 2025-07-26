#!/usr/bin/env python
import os
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from celeb import Celeb
from cub_voc import CUB_VOC
from load_utils import load_state_dict_from_url
from newPad2d import newPad2d

IS_TRAIN = 1  # 0/1
IS_MULTI = 0
LAYERS = '13'
DATANAME = 'celeb'
NUM_CLASSES = 6 if IS_MULTI else 2
if DATANAME == 'celeb':
    NUM_CLASSES = 80

cub_file = '../../datasets/frac_dataset'
voc_file = '../../datasets/VOCdevkit/VOC2010/voc2010_crop'
celeb_file = '../../datasets/CelebA_Cropped/CelebA/'

save_path = './icCNN/basic_fmap/vgg/'
acc_path = './icCNN/basic_fmap/vgg/acc/'

log_path = './icCNN/vgg/'
dataset = '%s_vgg_%s_ori' % (LAYERS, DATANAME)
log_path = log_path + dataset + '/'

pretrain_model = None

BATCHSIZE = 64
LR = 0.000001
EPOCH = 200

__all__ = ['VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19']

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], }

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth', }


class VGG(nn.Module):
    def __init__(self, features, num_classes=2, cfg='D', init_weights=True):
        super(VGG, self).__init__()

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.pad2d = newPad2d(1)  # nn.ReplicationPad2d(1)
        self.cfg = cfg
        self.classifier = nn.Sequential(  # 分类器结构
            # fc6
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(0.5),
            # fc7
            nn.Linear(4096, 512), nn.ReLU(True), nn.Dropout(0.5),
            # fc8
            nn.Linear(512, num_classes))

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        target_layer = 42 if self.cfg == 'D' else 33
        f_map = None
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.Conv2d):
                x = self.pad2d(x)
            x = layer(x)
            if i == target_layer:
                f_map = x.detach()
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, f_map

    def _initialize_weights(self):
        for layer, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, 3, padding=0)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]  # new padding
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]  # new padding
            in_channels = v
    return nn.Sequential(*layers)


def vgg(arch, cfg, num_class, device=None, pretrained=False, progress=True, **kwargs):
    model = VGG(make_layers(cfgs[cfg], batch_norm=True), num_class, cfg, **kwargs)
    if pretrained:
        pretrain_layer = 39 if cfg == 'D' else 30
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        pretrained_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k and int(k.split('.')[1]) <= pretrain_layer}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        device = torch.device("cuda")
        # model = nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(pretrain_model))
    return model


def get_Data(is_train, dataset_name, batch_size):
    transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    celeb_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    voc_helen_name = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'helen', 'voc_multi']
    label = None if is_train else 0
    if dataset_name == 'cub':
        trainset = CUB_VOC(cub_file, dataset_name, 'ori', train=True, transform=transform, is_frac=label)
        testset = CUB_VOC(cub_file, dataset_name, 'ori', train=False, transform=val_transform, is_frac=label)
    # cropped voc dataset
    elif dataset_name in voc_helen_name:
        trainset = CUB_VOC(voc_file, dataset_name, 'ori', train=True, transform=transform, is_frac=label)
        testset = CUB_VOC(voc_file, dataset_name, 'ori', train=False, transform=val_transform, is_frac=label)
    # celeba dataset
    elif dataset_name == 'celeb':
        trainset = Celeb(celeb_file, training = True, transform=celeb_transform, train_num=10240)
        testset = Celeb(celeb_file, training = False, transform=celeb_transform, train_num=19962)
    print(len(trainset), len(testset))
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


def net_train():
    trainset_loader, testset_loader = get_Data(IS_TRAIN, DATANAME, BATCHSIZE)
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
        os.makedirs(log_path)
    else:
        os.makedirs(log_path)
    device = torch.device("cuda")

    layer_arch = 'vgg13_bn' if LAYERS == '13' else 'vgg16_bn'
    layer_cfg = 'B' if LAYERS == '13' else 'D'
    net = vgg(arch=layer_arch, cfg=layer_cfg, num_class=NUM_CLASSES, device=device, pretrained=True, progress=True, ).cuda()
    # net = nn.DataParallel(net).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=125, gamma=0.6)
    test = test_celeb if DATANAME == 'celeb' else test_ori

    # Train the model
    best_acc = 0.0;
    save_loss = [];
    for epoch in range(0, EPOCH + 1):
        scheduler.step()
        net.train()
        total_loss = 0.0;
        for batch_step, input_data in tqdm(enumerate(trainset_loader, 0), total=len(trainset_loader), smoothing=0.9):
            inputs, labels = input_data
            inputs, labels = inputs.to(device), labels.to(device).long()
            optimizer.zero_grad()
            output, _ = net(inputs)
            if DATANAME != 'celeb':
                loss = criterion(output, labels)
            else:
                loss = .0
                for attribution in range(NUM_CLASSES // 2):
                    loss += criterion(output[:, 2 * attribution:2 * attribution + 2], labels[:, attribution])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        ### loss save code #####
        total_loss = float(total_loss) / len(trainset_loader)
        save_loss.append(total_loss)
        np.savez(os.path.join(log_path, 'loss.npz'), loss=np.array(save_loss))
        if epoch % 50 == 0 or epoch == 0:
            train_acc = test(net, trainset_loader, NUM_CLASSES)
            print('Epoch', epoch, 'train accuracy:%.4f' % train_acc)
            torch.save(net.state_dict(), log_path + '/model_%.3d.pth' % (epoch))
    print('Finished Training')
    return net


def get_feature():
    _, testset_test = get_Data(True, DATANAME, BATCHSIZE)
    _, testset_feature = get_Data(False, DATANAME, BATCHSIZE)
    device = torch.device("cuda")
    layer_arch = 'vgg13_bn' if LAYERS == '13' else 'vgg16_bn'
    layer_cfg = 'B' if LAYERS == '13' else 'D'

    if not os.path.exists(pretrain_model):
        raise Exception("Not such pretrain-model!")
    net = vgg(arch=layer_arch, cfg=layer_cfg, num_class=NUM_CLASSES, device=device, pretrained=False, progress=True, )
    # net = nn.DataParallel(net).to(device)
    test = test_celeb if DATANAME == 'celeb' else test_ori
    acc = test(net, testset_test, NUM_CLASSES)
    print('test acc:', acc)
    # if not os.path.exists(acc_path):
    #    os.makedirs(acc_path)
    f = open(os.path.join(acc_path, layer_arch + '_' + DATANAME + '_test.txt'), 'w+')
    f.write('%s\n' % layer_arch)
    f.write('%s\n' % DATANAME)
    f.write('acc:%f\n' % acc)
    # if not os.path.exists(save_path):
    #    os.makedirs(save_path)
    testset = testset_test if DATANAME == 'voc_multi' else testset_feature
    all_feature = []
    for batch_step, input_data in tqdm(enumerate(testset, 0), total=len(testset), smoothing=0.9):
        inputs, labels = input_data
        inputs, labels = inputs.cuda(), labels.cuda()
        net.eval()
        _, f_map = net(inputs)
        all_feature.append(f_map.cpu().numpy())
    all_feature = np.concatenate(all_feature, axis=0)
    f.write('sample num:%d' % (all_feature.shape[0]))
    f.close()
    print(all_feature.shape)
    np.savez_compressed(save_path + LAYERS + '_vgg_' + DATANAME + '_ori.npz', f_map=all_feature[...])
    print('Finished Getting Feature!')
    return net


def test_ori(net, testdata, n_cls):
    correct, total = .0, .0
    for inputs, labels in testdata:
        inputs, labels = inputs.cuda(), labels.cuda().long()
        net.eval()
        outputs, _ = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return float(correct) / total


def test_celeb(net, testdata, n_cls):
    correct, total = .0, .0
    ATTRIBUTION_NUM = n_cls // 2
    running_correct = np.zeros(ATTRIBUTION_NUM)
    for inputs, labels in tqdm(testdata):
        inputs, labels = inputs.cuda(), labels.cuda().long()
        net.eval()
        outputs, _ = net(inputs)
        out = outputs.data
        total += labels.size(0)
        for attribution in range(ATTRIBUTION_NUM):
            _, predicted = torch.max(out[:, 2 * attribution:2 * attribution + 2], 1)
            correct = (predicted == labels[:, attribution]).sum().item()
            running_correct[attribution] += correct
    attr_acc = running_correct / float(total)
    return np.mean(attr_acc)


if __name__ == '__main__':

    random.seed(1)
    os.environ['PYTHONHASHSEED'] = str(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    if IS_TRAIN:
        net = net_train()
    else:
        net = get_feature()
