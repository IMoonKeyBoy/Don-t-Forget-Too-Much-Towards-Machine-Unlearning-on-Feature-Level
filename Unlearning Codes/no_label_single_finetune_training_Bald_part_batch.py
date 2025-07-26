import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from cprint import cprint
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.augmentations import get_transforms_adv
from dataset.dataset import CelebASingleClassifiation
from functions import make_grid
from no_label_single_finetune_model_mouth import ObjectMultiLabelAdv


def addTransparency(img, factor=0):
    img = img.convert('RGBA')
    img_blender = Image.new('RGBA', img.size, (0, 0, 0, 0))
    img = Image.blend(img_blender, img, factor)
    return img


def get_cluster(matrix):
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
    for i, channels in enumerate(cluser):
        print('Group', i, 'contains', len(channels), 'channels.')
    return cluser


def channel_max_min_whole(f_map):
    T, C, H, W = f_map.shape
    max_v = np.max(f_map, axis=(0, 2, 3), keepdims=True)
    min_v = np.min(f_map, axis=(0, 2, 3), keepdims=True)
    return (f_map - min_v) / (max_v - min_v + 1e-6)


def main(args):
    # Dataset

    train_transforms = get_transforms_adv(args.img_size, mode='train')
    valid_transforms = get_transforms_adv(args.img_size, mode='valid')

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

    trainset_loader = DataLoader(train_dataset,
                                 args.batch_size,
                                 drop_last=True,
                                 num_workers=6,
                                 pin_memory=True)
    testset_loader = DataLoader(valid_dataset,
                                args.batch_size,
                                drop_last=True,
                                num_workers=6,
                                pin_memory=True)

    model = ObjectMultiLabelAdv(args)

    if args.multi:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        device = torch.device("cuda:0")
        model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1])
    else:
        model = model.cuda()

    def trainable_params():
        for param in model.parameters():
            if param.requires_grad:
                yield param

    optimizer = torch.optim.Adam(trainable_params(), args.lr, weight_decay=1e-06)

    if args.wandb:
        wandb.init(project='Feature unlearning',
                   group=args.group_name,
                   name=args.run_name, config=args)
        wandb.watch(model)

    for epoch in range(args.start_epoch, args.epoch):
        training_logs = {}
        encode_adv_acc, encoded_images = train(epoch, model, trainset_loader, optimizer=optimizer,
                                               testset_loader=testset_loader)

        training_logs["Valid encode_adv_acc"] = encode_adv_acc
        encoded_images = wandb.Image(
            make_grid(encoded_images.cpu(), (int(len(encoded_images) ** 0.5), int(len(encoded_images) ** 0.5))),
            caption="epoch:{}".format(epoch))
        training_logs["Training Auto_encoded_images"] = encoded_images
        task_accuracy, outputs_images = test(epoch, model, testset_loader)
        training_logs["Valid Task_accuracy"] = task_accuracy
        outputs_images = wandb.Image(
            make_grid(outputs_images.cpu(), (int(len(outputs_images) ** 0.5), int(len(outputs_images) ** 0.5))),
            caption="epoch:{}".format(epoch))
        training_logs["Valid Auto_encoded_images"] = outputs_images
        if args.wandb:
            wandb.log(training_logs)



def train(epoch, model, train_loader, optimizer, testset_loader):
    adv_encode_preds = []
    adv_truth = []

    for batch_idx, sample in enumerate(train_loader):
        model.train()
        imgs, labels = sample['imgs'].float().cuda(), sample['labels'].float().cuda()
        encode_pros, outputs_images = model(imgs, my_eval=False)

        loss = F.cross_entropy(encode_pros, labels.max(1, keepdim=False)[1], reduction='mean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        encode_pros = np.argmax(F.softmax(encode_pros, dim=1).cpu().detach().numpy(), axis=1)
        adv_encode_preds += encode_pros.tolist()
        adv_truth += labels.cpu().max(1, keepdim=False)[1].numpy().tolist()
        encode_adv_acc = accuracy_score(adv_truth, adv_encode_preds)

        adv_acc, temp = test(epoch, model, testset_loader)
        training_logs = {}
        training_logs["Valid Encode_adv_acc"] = adv_acc
        encoded_images = wandb.Image(make_grid(outputs_images.cpu(), (int(len(outputs_images) ** 0.5), int(len(outputs_images) ** 0.5))), caption="batch_idx:{}".format(batch_idx))
        training_logs["Valid Auto_encoded_images"] = encoded_images
        if args.wandb:
            wandb.log(training_logs)

    encode_adv_acc = accuracy_score(adv_truth, adv_encode_preds)
    return encode_adv_acc, outputs_images


def test(epoch, model, testset_loader):
    adv_preds = []
    adv_truth = []
    model.train()
    for batch_idx, sample in enumerate(testset_loader):
        imgs, labels = sample['imgs'].float().cuda(), sample['labels'].float().cuda()
        probs, outputs_images = model(imgs, my_eval=True)
        probs = np.argmax(F.softmax(probs, dim=1).cpu().detach().numpy(), axis=1)
        adv_preds += probs.tolist()
        adv_truth += labels.cpu().max(1, keepdim=False)[1].numpy().tolist()
    adv_acc = accuracy_score(adv_truth, adv_preds)
    return adv_acc, outputs_images


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-03)
    parser.add_argument('--img_size', type=int, default=224, help='Size of input image')
    parser.add_argument('--model', type=str, default='customresnet',
                        choices=('customresnet', 'customefficientnet', 'effnetv2s'))

    parser.add_argument('--run_name', type=str, default="Unlearning Bird backgroud")
    parser.add_argument('--group_name', type=str, default="Adversary Training Process Without Label")
    parser.add_argument('--wandb', default=True, action='store_true', help='Use wandb for logging')
    parser.add_argument('--multi', default=False, action='store_true', help='Use wandb for logging')

    parser.add_argument('--data_path', type=str, default='../../datasets/CelebaK/', help='Root path of data')
    parser.add_argument('--task_labels', type=list, default=['Bald'], help='Attributes')

    # High_Cheekbones young
    args = parser.parse_args()
    args.pretrained_network_path = '../icCNN-main-new/icCNN/resnet/18_resnet_celeb_iccnn_59/model_2500.pth'
    args.original_model = './checkpoints/original/' + args.task_labels[0] + "/" + args.task_labels[0] + '_final.pth'
    args.run_name = args.task_labels[0] +"_"+ str(args.lr)
    for k, v in sorted(vars(args).items()):
        cprint.info(k, '=', v)
    main(args)
