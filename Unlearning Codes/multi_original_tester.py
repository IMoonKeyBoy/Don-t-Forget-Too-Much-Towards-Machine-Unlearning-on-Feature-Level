import argparse
import os
import random

import numpy as np
import torch
import wandb
from cprint import cprint
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.augmentations import get_transforms
from dataset.dataset import CelebAMultiClassifiation
from functions import get_optimizer, get_scheduler, get_lr, get_multi_classification_model
import wandb
import argparse
import copy
import os
import random
from functions import AverageMeter, make_grid, get_lr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from multi_ae_adv_finetuning_model import ObjectMultiLabelAdv
from cprint import cprint
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm as tqdm
# from logger import Logger
from functions import get_partition

from dataset.augmentations import get_transforms_adv
from dataset.dataset import ADV_CelebAMultiClassifiation


def train_one_epoch(args,
                    epoch,
                    device,
                    model,
                    loader,
                    optimizer,
                    criterion):
    model.train()
    train_loss_list = list()
    train_accuracy = []
    for i in range(len(args.task)):
        train_accuracy.append([])

    pbar = tqdm(loader, total=loader.__len__(), position=0, leave=True)
    for sample in pbar:
        optimizer.zero_grad()
        imgs, labels = sample['imgs'].float().to(device), sample['labels'].float().to(device)

        probs = model(imgs)

        loss = criterion(probs, labels)
        loss.backward()
        optimizer.step()

        probs, labels = probs.cpu().detach().numpy(), labels.cpu().detach().numpy()

        train_loss = loss.item()
        train_loss_list.append(train_loss)

        for i in range(len(args.task)):
            preds_index = probs[:, i] > 0.5
            preds_acc = (labels[:, i] == preds_index).mean()
            train_accuracy[i].append(preds_acc)

        # refresh the show bar
        desc = f"Train Epoch: {epoch + 1}, Loss: {np.mean(train_loss_list):.2f}"
        for i in range(len(args.task)):
            desc = desc + ", " + args.task[i] + f": {np.mean(train_accuracy[i]):.2f}"
        pbar.set_description(desc)

    return np.mean(train_loss_list), train_accuracy


def valid_one_epoch(args,
                    epoch,
                    device,
                    model,
                    loader,
                    criterion):
    valid_loss_list = list()
    model.eval()

    valid_accuracy = []
    for i in range(len(args.target)):
        valid_accuracy.append([])

    pbar = tqdm(loader, total=loader.__len__(), position=0, leave=True)
    for sample in pbar:
        imgs, task, labels = sample['imgs'].float().cuda(), sample['task_labels'].float().cuda(), sample['target_labels'].float().cuda()
        probs = model(imgs)
        loss = criterion(probs, labels)

        probs, labels = probs.cpu().detach().numpy(), labels.cpu().detach().numpy()

        valid_loss = loss.item()
        valid_loss_list.append(valid_loss)

        for i in range(len(args.target)):
            preds_index = probs[:, i] > 0.5
            preds_acc = (labels[:, i] == preds_index).mean()
            valid_accuracy[i].append(preds_acc)

        # refresh the show bar
        desc = f"Valid Epoch: {epoch + 1}, Loss : {np.mean(valid_loss_list):.2f}"
        for i in range(len(args.target)):
            desc = desc + ", " + args.target[i]  + f": {np.mean(valid_accuracy[i]):.2f}"
        pbar.set_description(desc)

    return np.mean(valid_loss_list), valid_accuracy


def training(args):
    '''
    args.save_path = os.path.join(args.save_path, args.run_name)

    if os.path.exists(args.save_path):
        cprint.err("Folder is exiting! Enter your action: 1.exit, 2.continue")
        choose = input()
        if choose == "1":
            return
        elif choose == "2":
            pass
    else:
        os.makedirs(args.save_path)

    train_transforms = get_transforms(args.img_size, mode='train')
    valid_transforms = get_transforms(args.img_size, mode='valid')

    train_dataset = CelebAMultiClassifiation(args.data_path, args.task, transforms=train_transforms, is_training=True)
    valid_dataset = CelebAMultiClassifiation(args.data_path, args.task, transforms=valid_transforms, is_training=False)

    train_loader = DataLoader(train_dataset, args.batch_size, drop_last=True, num_workers=6, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size, drop_last=False, num_workers=6, pin_memory=True)

    '''
    valid_transforms = get_transforms_adv(args.img_size, mode='valid')

    sub_dataset_test = ADV_CelebAMultiClassifiation(args.data_path,
                                                    args.task,
                                                    args.target,
                                                    transforms=valid_transforms,
                                                    is_split_notask_and_task_dataset=True,
                                                    is_training=False,
                                                    is_task=True)

    valid_loader = torch.utils.data.DataLoader(sub_dataset_test,
                                             batch_size=args.batch_size,
                                             drop_last=True,
                                             num_workers=6,
                                             shuffle=True)

    print(len(sub_dataset_test),len(valid_loader))
    model = get_multi_classification_model(args.model, len(args.task))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.wandb:
        wandb.init(project='Feature unlearning',
                   group=args.group_name,
                   name=args.run_name, config=args)
        wandb.watch(model)

    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args.scheduler, optimizer)
    criterion = torch.nn.BCELoss()

    pretrained_adversary_network_path = os.path.join('./checkpoints/original_multi/', '_'.join(args.target), '_'.join(args.target) + "_final.pth")
    model = torch.load(pretrained_adversary_network_path)
    with torch.no_grad():
        valid_loss, valid_accuracies = valid_one_epoch(args=args,
                                                       epoch=0,
                                                       device=device,
                                                       model=model,
                                                       loader=valid_loader,
                                                       criterion=criterion)
    for i in range(len(args.target)):
        print(args.target[i], np.mean(valid_accuracies[i]))


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
    parser.add_argument('--save_path', type=str, default='./checkpoints/original_multi/', help='Dir path to save model weights')

    parser.add_argument('--epoch', type=int, default=10, help='Total Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-06, help='Learning Rate')
    parser.add_argument('--img_size', type=int, default=224, help='Size of input image')
    parser.add_argument('--model', type=str, default='customresnet', choices=('customresnet', 'customefficientnet', 'effnetv2s'))
    parser.add_argument('--optimizer', type=str, default='adamw', choices=('adam', 'adamw'))
    parser.add_argument('--weight_decay', type=float, default=1e-06)
    parser.add_argument('--scheduler', type=str, default='cosinewarmup', choices=('none', 'cosinewarmup'))

    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--training', default=False, action='store_true', help='Use wandb for logging')

    parser.add_argument('--wandb', default=True, action='store_true', help='Use wandb for logging')
    parser.add_argument('--group_name', type=str, default='Multi Classification Training Process', choices=('none', 'cosinewarmup'))

    parser.add_argument('--target', type=list, default=['Mouth_Slightly_Open','Bald'], help='Attributes')
    parser.add_argument('--task', type=list, default=['Bald'], help='Attributes')


    args = parser.parse_args()
    args.run_name = '_'.join(args.task)

    for k, v in sorted(vars(args).items()):
        cprint.info(k, '=', v)

    training(args)
