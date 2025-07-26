import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from cprint import cprint
from sklearn.metrics import accuracy_score  # 预测精度
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm

from dataset.augmentations import get_transforms_adv
from dataset.dataset import ADV_CelebASingleClassifiation
from functions import AverageMeter, make_grid
from single_adversary_model import ObjectMultiLabelAdv


def main(args):
    if os.path.exists(args.save_path + "/" + args.run_name):
        cprint.info('Path {} exists! and not resuming'.format(args.save_path + "/" + args.run_name))
    if not os.path.exists(args.save_path + "/" + args.run_name):
        os.makedirs(args.save_path + "/" + args.run_name)

    train_transforms = get_transforms_adv(args.img_size, mode='train')

    sub_dataset_training_is_task = ADV_CelebASingleClassifiation(args.data_path,
                                                                 args.task,
                                                                 args.target,
                                                                 transforms=train_transforms,
                                                                 is_split_notask_and_task_dataset=True,
                                                                 is_training=True,
                                                                 is_task=True)

    sub_dataset_training = ADV_CelebASingleClassifiation(args.data_path,
                                                         args.task,
                                                         args.target,
                                                         transforms=train_transforms,
                                                         is_split_notask_and_task_dataset=True,
                                                         is_training=True,
                                                         is_task=True)

    sub_dataset_test = ADV_CelebASingleClassifiation(args.data_path,
                                                     args.task,
                                                     args.target,
                                                     transforms=train_transforms,
                                                     is_split_notask_and_task_dataset=True,
                                                     is_training=False,
                                                     is_task=True)

    train_loader_is_task = torch.utils.data.DataLoader(sub_dataset_training_is_task,
                                                       batch_size=args.batch_size,
                                                       drop_last=True,
                                                       num_workers=6,
                                                       shuffle=False)

    train_loader = torch.utils.data.DataLoader(sub_dataset_training,
                                               batch_size=args.batch_size,
                                               drop_last=True,
                                               num_workers=6,
                                               shuffle=False)

    val_loader = torch.utils.data.DataLoader(sub_dataset_test,
                                             batch_size=args.batch_size,
                                             drop_last=True,
                                             num_workers=6,
                                             shuffle=True)

    model = ObjectMultiLabelAdv(args, args.adv_lambda)
    if args.multi:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        device = torch.device("cuda:0")
        model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1])
    else:
        model = model.cuda()

    criterion_unet = torch.nn.L1Loss(reduction='mean')

    def trainable_params():
        for param in model.parameters():
            if param.requires_grad:
                yield param

    optimizer = torch.optim.Adam(trainable_params(), args.lr, weight_decay=1e-06)

    if args.wandb:
        wandb.init(project='Feature unlearning',
                   group=args.group_name,
                   name=args.run_name + '_' + str(args.adv_lambda) + '_' + str(args.beta), config=args)
        wandb.watch(model)

    if args.checkpoint is not None:
        if os.path.isfile(os.path.join(args.save_dir, os.path.join(args.save_path, args.run_name + '_' + str(args.adv_lambda) + '_' + str(args.beta) + '_checkpoint_%d.pth.tar' % args.checkpoint))):
            cprint.warn("=> loading checkpoint '{}'".format(args.save_dir))
            checkpoint = torch.load(os.path.join(args.save_path, args.run_name + '_' + str(args.adv_lambda) + '_' + str(args.beta) + '_checkpoint_%d.pth.tar' % args.checkpoint))
            args.start_epoch = checkpoint['epoch']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            cprint.warn("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            cprint.warn("=> no checkpoint found at '{}'".format(args.save_dir))
    else:
        cprint.warn("Adversary Training From Scratch")

    for epoch in range(args.start_epoch, args.epoch):

        training_logs = {}
        model = model.module
        model.training_autoencoder()
        model = model.cuda()

        def trainable_params_new():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    yield param

        optimizer = torch.optim.Adam(trainable_params_new(), args.lr, weight_decay=1e-06)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1])

        task_accuracy, target_accuracy, task_model_loss, target_model_loss, unet_model_loss_logger, autoencoded_images = test(epoch,
                                                                                                                              model,
                                                                                                                              criterion_unet,
                                                                                                                              val_loader)

        training_logs["Valid Task_accuracy"] = task_accuracy
        training_logs["Valid Target_accuracy"] = target_accuracy
        autoencoded_images = wandb.Image(make_grid(autoencoded_images.cpu(), (6, 6)), caption="epoch:{}".format(epoch))
        training_logs["Valid Auto_encoded_images"] = autoencoded_images

        task_accuracy, target_accuracy, task_model_loss, target_model_loss, unet_model_loss_logger, autoencoded_images = train(epoch,
                                                                                                                               model,
                                                                                                                               criterion_unet,
                                                                                                                               train_loader_is_task,
                                                                                                                               optimizer)

        training_logs["Training Task_accuracy"] = task_accuracy
        training_logs["Training Target_accuracy"] = target_accuracy
        training_logs["Training Task_model_loss"] = task_model_loss
        training_logs["Training Target_model_loss"] = target_model_loss
        training_logs["Training Unet_model_loss_logger"] = unet_model_loss_logger
        autoencoded_images = wandb.Image(make_grid(autoencoded_images.cpu(), (6, 6)), caption="epoch:{}".format(epoch))
        training_logs["Training Auto_encoded_images"] = autoencoded_images

        model = model.module
        model.training_original()

        def trainable_params_new():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    yield param

        model = model.cuda()
        optimizer = torch.optim.Adam(trainable_params_new(), args.lr, weight_decay=1e-06)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1])

        task_accuracy, task_model_loss, autoencoded_images = finetuning_train(epoch,
                                                                              model,
                                                                              criterion_unet,
                                                                              train_loader,
                                                                              optimizer)
        training_logs["Finetuning_ori Training Task_accuracy"] = task_accuracy
        autoencoded_images = wandb.Image(make_grid(autoencoded_images.cpu(), (6, 6)), caption="epoch:{}".format(epoch))
        training_logs["Finetuning_ori Training Auto_encoded_images"] = autoencoded_images

        task_accuracy, target_accuracy, task_model_loss, target_model_loss, unet_model_loss_logger, autoencoded_images = finetuning_test(epoch,
                                                                                                                                         model,
                                                                                                                                         criterion_unet,
                                                                                                                                         val_loader)

        training_logs["Finetuning_ori Valid Task_accuracy"] = task_accuracy
        training_logs["Finetuning_ori Valid Target_accuracy"] = target_accuracy
        autoencoded_images = wandb.Image(make_grid(autoencoded_images.cpu(), (6, 6)), caption="epoch:{}".format(epoch))
        training_logs["Finetuning_ori Valid Auto_encoded_images"] = autoencoded_images

        torch.save(model.state_dict(), args.save_path + "/" + args.run_name + "/" + args.run_name + "_" + str(epoch) + "_.pt")

        if args.wandb:
            wandb.log(training_logs)


def finetuning_train(epoch, model, criterion_unet, train_loader, optimizer):
    model.train()
    task_loss_logger = AverageMeter()

    task_preds = []
    task_truths = []

    for batch_idx, sample in enumerate(tqdm(train_loader, desc='Val %d' % epoch)):
        images, task, target = sample['imgs'].float().cuda(), sample['task_labels'].float().cuda(), sample['target_labels'].float().cuda()

        task_pred, target_pred, auto_encoded_images = model(images)
        task_loss = F.cross_entropy(task_pred, task.max(1, keepdim=False)[1], reduction='mean')
        task_pred = np.argmax(F.softmax(task_pred, dim=1).cpu().detach().numpy(), axis=1)
        task_preds += task_pred.tolist()
        task_truths += task.cpu().max(1, keepdim=False)[1].numpy().tolist()

        task_loss_logger.update(task_loss.item())

        optimizer.zero_grad()
        task_loss.backward()
        optimizer.step()

    task_accuracy = accuracy_score(task_truths, task_preds)
    return task_accuracy, task_loss_logger.avg, auto_encoded_images[0:36]


def finetuning_test(epoch, model, criterion_unet, val_loader):
    model.eval()
    task_loss_logger = AverageMeter()
    adv_loss_logger = AverageMeter()
    l1_loss_logger = AverageMeter()

    task_preds = []
    task_truths = []

    target_preds = []
    target_truths = []

    for batch_idx, sample in enumerate(tqdm(val_loader, desc='Val %d' % epoch)):
        images, task, target = sample['imgs'].float().cuda(), sample['task_labels'].float().cuda(), sample['target_labels'].float().cuda()

        # Forward, Backward and Optimizer
        task_pred, target_pred, auto_encoded_images = model(images)
        unet_loss = criterion_unet(auto_encoded_images, images)

        task_loss = F.cross_entropy(task_pred, task.max(1, keepdim=False)[1], reduction='mean')
        task_pred = np.argmax(F.softmax(task_pred, dim=1).cpu().detach().numpy(), axis=1)
        task_preds += task_pred.tolist()
        task_truths += task.cpu().max(1, keepdim=False)[1].numpy().tolist()

        adv_loss = F.cross_entropy(target_pred, target.max(1, keepdim=False)[1], reduction='mean')
        target_pred = np.argmax(F.softmax(target_pred, dim=1).cpu().detach().numpy(), axis=1)
        target_preds += target_pred.tolist()
        target_truths += target.cpu().max(1, keepdim=False)[1].numpy().tolist()

        task_loss_logger.update(task_loss.item())
        adv_loss_logger.update(adv_loss.item())
        l1_loss_logger.update(unet_loss.item())

        if batch_idx == 200:
            break
    task_accuracy = accuracy_score(task_truths, task_preds)
    target_accuracy = accuracy_score(target_truths, target_preds)
    return task_accuracy, target_accuracy, task_loss_logger.avg, adv_loss_logger.avg, l1_loss_logger.avg, auto_encoded_images[0:36]


def train(epoch, model, criterion_unet, train_loader, optimizer):
    model.train()
    task_loss_logger = AverageMeter()
    target_loss_logger = AverageMeter()
    unet_loss_logger = AverageMeter()

    task_preds = []
    task_truths = []

    target_preds = []
    target_truths = []

    for batch_idx, sample in enumerate(tqdm(train_loader, desc='Train %d' % epoch)):
        images, task, target = sample['imgs'].float().cuda(), sample['task_labels'].float().cuda(), sample['target_labels'].float().cuda()

        task_pred, target_pred, auto_encoded_images = model(images)
        unet_loss = criterion_unet(auto_encoded_images, images)

        task_loss = F.cross_entropy(task_pred, task.max(1, keepdim=False)[1], reduction='mean')
        task_pred = np.argmax(F.softmax(task_pred, dim=1).cpu().detach().numpy(), axis=1)
        task_preds += task_pred.tolist()
        task_truths += task.cpu().max(1, keepdim=False)[1].numpy().tolist()

        adv_loss = F.cross_entropy(target_pred, target.max(1, keepdim=False)[1], reduction='mean')
        target_pred = np.argmax(F.softmax(target_pred, dim=1).cpu().detach().numpy(), axis=1)
        target_preds += target_pred.tolist()
        target_truths += target.cpu().max(1, keepdim=False)[1].numpy().tolist()

        task_loss_logger.update(task_loss.item())
        target_loss_logger.update(adv_loss.item())
        unet_loss_logger.update(unet_loss.item())
        loss = task_loss + adv_loss + args.beta * unet_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    task_accuracy = accuracy_score(task_truths, task_preds)
    target_accuracy = accuracy_score(target_truths, target_preds)

    return task_accuracy, target_accuracy, task_loss_logger.avg, target_loss_logger.avg, unet_loss_logger.avg, auto_encoded_images[0:36]


def test(epoch, model, criterion_unet, val_loader):
    model.eval()
    task_loss_logger = AverageMeter()
    adv_loss_logger = AverageMeter()
    l1_loss_logger = AverageMeter()

    task_preds = []
    task_truths = []

    target_preds = []
    target_truths = []

    for batch_idx, sample in enumerate(tqdm(val_loader, desc='Val %d' % epoch)):
        images, task, target = sample['imgs'].float().cuda(), sample['task_labels'].float().cuda(), sample['target_labels'].float().cuda()
        task_pred, target_pred, auto_encoded_images = model(images)

        unet_loss = criterion_unet(auto_encoded_images, images)

        task_loss = F.cross_entropy(task_pred, task.max(1, keepdim=False)[1], reduction='mean')
        task_pred = np.argmax(F.softmax(task_pred, dim=1).cpu().detach().numpy(), axis=1)
        task_preds += task_pred.tolist()
        task_truths += task.cpu().max(1, keepdim=False)[1].numpy().tolist()

        adv_loss = F.cross_entropy(target_pred, target.max(1, keepdim=False)[1], reduction='mean')
        target_pred = np.argmax(F.softmax(target_pred, dim=1).cpu().detach().numpy(), axis=1)
        target_preds += target_pred.tolist()
        target_truths += target.cpu().max(1, keepdim=False)[1].numpy().tolist()

        task_loss_logger.update(task_loss.item())
        adv_loss_logger.update(adv_loss.item())
        l1_loss_logger.update(unet_loss.item())

        if batch_idx == 200:
            break

    task_accuracy = accuracy_score(task_truths, task_preds)
    target_accuracy = accuracy_score(target_truths, target_preds)
    return task_accuracy, target_accuracy, task_loss_logger.avg, adv_loss_logger.avg, l1_loss_logger.avg, auto_encoded_images[0:36]


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../datasets/CelebaK/', help='Root path of data')
    parser.add_argument('--save_path', type=str, default='./checkpoints/singleunlearning/', help='path for saving checkpoints')

    parser.add_argument('--adv_on', action='store_true', default=True, help='start adv training')
    parser.add_argument('--adv_lambda', type=float, default=5.0, help='weight assigned to adv loss')
    parser.add_argument('--beta', type=float, default=5.0, help='autoencoder l1 loss weight')
    parser.add_argument('--autoencoder_finetune', action='store_true', default=True)
    parser.add_argument('--finetune', action='store_true')

    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--lr', type=float, default=5e-06)
    parser.add_argument('--img_size', type=int, default=224, help='Size of input image')
    parser.add_argument('--model', type=str, default='customresnet', choices=('customresnet', 'customefficientnet', 'effnetv2s'))

    parser.add_argument('--pretrained_task_network_path', type=str, default='./checkpoints/original/')
    parser.add_argument('--pretrained_target_network_path', type=str, default='./checkpoints/original/')

    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--group_name', type=str, default="Adversary training and unlearning for single Finally")
    parser.add_argument('--wandb', default=True, action='store_true', help='Use wandb for logging')

    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--training', default=True, action='store_true', help='Use wandb for logging')
    parser.add_argument('--multi', default=True, action='store_true', help='Use wandb for logging')

    parser.add_argument('--task', type=list, default=['Bald'], help='Attributes')
    parser.add_argument('--target', type=list, default=['Mouth_Slightly_Open'], help='Attributes')

    args = parser.parse_args()
    args.pretrained_task_network_path = os.path.join(args.pretrained_task_network_path, '_'.join(args.task), '_'.join(args.task) + "_final.pth")
    args.pretrained_target_network_path = os.path.join(args.pretrained_target_network_path, '_'.join(args.target), '_'.join(args.target) + "_final.pth")

    args.run_name = '_'.join(args.task) + "_" + '_'.join(args.target)
    args.save_path = os.path.join(args.save_path, args.run_name + "_" + str(args.adv_lambda) + '_' + str(args.beta))

    for k, v in sorted(vars(args).items()):
        cprint.info(k, '=', v)

    main(args)
