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


def main(args):
    '''
    if os.path.exists(args.save_path):
        cprint.err("Folder is exiting! Enter your action: 1.exit, 2.continue")
        choose = input()
        if choose == "1":
            return
        elif choose == "2":
            pass
    else:
        os.makedirs(args.save_path)
    '''
    if os.path.exists(args.save_path + "/" + args.run_name):
        cprint.info('Path {} exists! and not resuming'.format(args.save_path + "/" + args.run_name))
    if not os.path.exists(args.save_path + "/" + args.run_name):
        os.makedirs(args.save_path + "/" + args.run_name)

    train_transforms = get_transforms_adv(args.img_size, mode='train')
    valid_transforms = get_transforms_adv(args.img_size, mode='valid')

    sub_dataset_training_is_task = ADV_CelebAMultiClassifiation(args.data_path,
                                                                args.task,
                                                                args.target,
                                                                transforms=train_transforms,
                                                                is_split_notask_and_task_dataset=True,
                                                                is_training=True,
                                                                is_task=True)

    sub_dataset_training = ADV_CelebAMultiClassifiation(args.data_path,
                                                        args.task,
                                                        args.target,
                                                        transforms=train_transforms,
                                                        is_split_notask_and_task_dataset=True,
                                                        is_training=True,
                                                        is_task=True)

    sub_dataset_test = ADV_CelebAMultiClassifiation(args.data_path,
                                                    args.task,
                                                    args.target,
                                                    transforms=valid_transforms,
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
        model = ObjectMultiLabelAdv(args, args.adv_lambda).cuda()

    # load pretrained linear layer from the original model
    criterion = nn.BCELoss()
    criterion_unet = torch.nn.L1Loss(reduction='elementwise_mean')

    def trainable_params():
        for param in model.parameters():
            if param.requires_grad:
                yield param

    optimizer = torch.optim.Adam(trainable_params(), args.lr, weight_decay=1e-6)

    ''' init wandb '''
    if args.wandb:
        wandb.init(project='Feature unlearning',
                   group=args.group_name,
                   name=args.run_name + '_' + str(args.adv_lambda) + '_' + str(args.beta), config=args)
        wandb.watch(model)

    if args.checkpoint is not None:
        if os.path.isfile(os.path.join(args.save_dir, os.path.join(args.save_path, args.run_name + '_' + str(
                args.adv_lambda) + '_' + str(args.beta) + '_checkpoint_%d.pth.tar' % args.checkpoint))):
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
        cprint.warn("Training From Scrath")

    for epoch in range(args.start_epoch, args.epoch):
        training_logs = {}

        model = model.module
        model.training_autoencoder()

        def trainable_params_new():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    yield param

        optimizer = torch.optim.Adam(trainable_params_new(), args.lr, weight_decay=1e-06)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1])

        task_accracy, target_accracy, original_model_loss, adv_model_loss, unet_model_loss_logger, autoencoded_images = test(
            args, epoch,
            model,
            criterion,
            criterion_unet,
            val_loader)

        for i in range(len(args.task)):
            training_logs["valid " + args.task[i] + " accuracy"] = np.mean(task_accracy[i])
            print(np.mean(task_accracy[i]))
        for i in range(len(args.target)):
            training_logs["valid " + args.target[i] + " accuracy"] = np.mean(target_accracy[i])
            print(np.mean(target_accracy[i]))
        autoencoded_images = wandb.Image(make_grid(autoencoded_images.cpu(), (6, 6)), caption="epoch:{}".format(epoch))
        training_logs["valid original_model_loss"] = original_model_loss
        training_logs["valid adv_model_loss"] = adv_model_loss
        training_logs["valid unet_model_loss_logger"] = unet_model_loss_logger
        training_logs["valid autoencoded_images"] = autoencoded_images

        task_accracy, target_accracy, original_model_loss, adv_model_loss, unet_model_loss_logger, autoencoded_images = train(
            args, epoch,
            model,
            criterion,
            criterion_unet,
            train_loader_is_task,
            optimizer)

        for i in range(len(args.task)):
            training_logs["training " + args.task[i] + " accuracy"] = np.mean(task_accracy[i])
        for i in range(len(args.target)):
            training_logs["training " + args.target[i] + " accuracy"] = np.mean(target_accracy[i])
        autoencoded_images = wandb.Image(make_grid(autoencoded_images.cpu(), (6, 6)), caption="epoch:{}".format(epoch))
        training_logs["training original_model_loss"] = original_model_loss
        training_logs["training adv_model_loss"] = adv_model_loss
        training_logs["training unet_model_loss_logger"] = unet_model_loss_logger
        training_logs["training autoencoded_images"] = autoencoded_images
        training_logs["Learning Rate"] = get_lr(optimizer)

        model = model.module
        model.training_original()

        def trainable_params_new():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    yield param

        optimizer = torch.optim.Adam(trainable_params_new(), args.lr, weight_decay=1e-06)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1])
        task_accracy, target_accracy, original_model_loss, adv_model_loss, unet_model_loss_logger, autoencoded_images = finetuningtest(
            args, epoch,
            model,
            criterion,
            criterion_unet,
            val_loader)

        for i in range(len(args.task)):
            training_logs["Finetuning valid " + args.task[i] + " accuracy"] = np.mean(task_accracy[i])
        for i in range(len(args.target)):
            training_logs["Finetuning valid " + args.target[i] + " accuracy"] = np.mean(target_accracy[i])
        autoencoded_images = wandb.Image(make_grid(autoencoded_images.cpu(), (6, 6)), caption="epoch:{}".format(epoch))
        training_logs["Finetuning valid original_model_loss"] = original_model_loss
        training_logs["Finetuning valid adv_model_loss"] = adv_model_loss
        training_logs["Finetuning valid unet_model_loss_logger"] = unet_model_loss_logger
        training_logs["Finetuning valid autoencoded_images"] = autoencoded_images

        task_accracy, target_accracy, original_model_loss, adv_model_loss, unet_model_loss_logger, autoencoded_images = finetuningtrain(
            args, epoch,
            model,
            criterion,
            criterion_unet,
            train_loader,
            optimizer)

        for i in range(len(args.task)):
            training_logs["Finetuning training " + args.task[i] + " accuracy"] = np.mean(task_accracy[i])
        for i in range(len(args.target)):
            training_logs["Finetuning training " + args.target[i] + " accuracy"] = np.mean(target_accracy[i])
        autoencoded_images = wandb.Image(make_grid(autoencoded_images.cpu(), (6, 6)), caption="epoch:{}".format(epoch))
        training_logs["Finetuning training original_model_loss"] = original_model_loss
        training_logs["Finetuning training adv_model_loss"] = adv_model_loss
        training_logs["Finetuning training unet_model_loss_logger"] = unet_model_loss_logger
        training_logs["Finetuning training autoencoded_images"] = autoencoded_images
        training_logs["Finetuning Learning Rate"] = get_lr(optimizer)

        if args.wandb:
            wandb.log(training_logs)
        torch.save(model.state_dict(), args.save_path + "/" + args.run_name + "/" + args.run_name +"_"+ str(epoch) + "_.pt")


def train(args, epoch, model, criterion, criterionL1, train_loader, optimizer):
    model.train()
    original_loss_logger = AverageMeter()
    adv_loss_logger = AverageMeter()
    unet_loss_logger = AverageMeter()

    task_accracy = []
    for i in range(len(args.task)):
        task_accracy.append([])

    target_accracy = []
    for i in range(len(args.target)):
        target_accracy.append([])

    for batch_idx, sample in enumerate(tqdm(train_loader, desc='Train %d' % epoch)):
        images, task, target = sample['imgs'].float().cuda(), sample['task_labels'].float().cuda(), sample['target_labels'].float().cuda()

        # Forward, Backward and Optimizer
        task_pred, target_pred, autoencoded_images = model(images)

        unet_loss = criterionL1(autoencoded_images, images)
        task_loss = F.cross_entropy(task_pred, task.max(1, keepdim=False)[1], reduction='elementwise_mean')
        adv_loss = criterion(target_pred, target)

        original_loss_logger.update(task_loss.item())
        adv_loss_logger.update(adv_loss.item())
        unet_loss_logger.update(unet_loss.item())

        task_pred, target_pred = task_pred.cpu().detach().numpy(), target_pred.cpu().detach().numpy()
        task, target = task.cpu().detach().numpy(), target.cpu().detach().numpy()

        for i in range(len(args.task)):
            preds_index = task_pred[:, i] > 0.5
            preds_acc = (task[:, i] == preds_index).mean()
            task_accracy[i].append(preds_acc)

        for i in range(len(args.target)):
            preds_index = target_pred[:, i] > 0.5
            preds_acc = (target[:, i] == preds_index).mean()
            target_accracy[i].append(preds_acc)

        # backpropagation
        loss = task_loss + adv_loss + args.beta * unet_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx == 500:
            break
    return task_accracy, target_accracy, original_loss_logger.avg, adv_loss_logger.avg, unet_loss_logger.avg, autoencoded_images[0:36]


def test(args, epoch, model, criterion, criterionL1, val_loader):
    model.eval()
    task_loss_logger = AverageMeter()
    adv_loss_logger = AverageMeter()
    l1_loss_logger = AverageMeter()

    task_accracy = []
    for i in range(len(args.task)):
        task_accracy.append([])

    target_accracy = []
    for i in range(len(args.target)):
        target_accracy.append([])

    for batch_idx, sample in enumerate(tqdm(val_loader, desc='Val %d' % epoch)):
        images, task, target = sample['imgs'].float().cuda(), sample['task_labels'].float().cuda(), sample['target_labels'].float().cuda()
        task_pred, target_pred, autoencoded_images = model(images)

        unet_loss = criterionL1(autoencoded_images, images)
        task_loss = F.cross_entropy(task_pred, task.max(1, keepdim=False)[1], reduction='elementwise_mean')
        adv_loss = criterion(target_pred, target)

        task_loss_logger.update(task_loss.item())
        adv_loss_logger.update(adv_loss.item())
        l1_loss_logger.update(unet_loss.item())

        task_pred, target_pred = task_pred.cpu().detach().numpy(), target_pred.cpu().detach().numpy()
        task, target = task.cpu().detach().numpy(), target.cpu().detach().numpy()

        for i in range(len(args.task)):
            preds_index = task_pred[:, i] > 0.5
            preds_acc = (task[:, i] == preds_index).mean()
            task_accracy[i].append(preds_acc)

        for i in range(len(args.target)):
            preds_index = target_pred[:, i] > 0.5
            preds_acc = (target[:, i] == preds_index).mean()
            target_accracy[i].append(preds_acc)
        if batch_idx == 500:
            break
    return task_accracy, target_accracy, task_loss_logger.avg, adv_loss_logger.avg, l1_loss_logger.avg, autoencoded_images[0:36]


def finetuningtrain(args, epoch, model, criterion, criterionL1, train_loader, optimizer):
    model.train()
    original_loss_logger = AverageMeter()
    adv_loss_logger = AverageMeter()
    unet_loss_logger = AverageMeter()

    task_accracy = []
    for i in range(len(args.task)):
        task_accracy.append([])

    target_accracy = []
    for i in range(len(args.target)):
        target_accracy.append([])

    for batch_idx, sample in enumerate(tqdm(train_loader, desc='Train %d' % epoch)):
        images, task, target = sample['imgs'].float().cuda(), sample['task_labels'].float().cuda(), sample['target_labels'].float().cuda()

        # Forward, Backward and Optimizer
        task_pred, target_pred, autoencoded_images = model(images)

        unet_loss = criterionL1(autoencoded_images, images)
        task_loss = F.cross_entropy(task_pred, task.max(1, keepdim=False)[1], reduction='elementwise_mean')
        adv_loss = criterion(target_pred, target)

        original_loss_logger.update(task_loss.item())
        adv_loss_logger.update(adv_loss.item())
        unet_loss_logger.update(unet_loss.item())

        task_pred, target_pred = task_pred.cpu().detach().numpy(), target_pred.cpu().detach().numpy()
        task, target = task.cpu().detach().numpy(), target.cpu().detach().numpy()

        for i in range(len(args.task)):
            preds_index = task_pred[:, i] > 0.5
            preds_acc = (task[:, i] == preds_index).mean()
            task_accracy[i].append(preds_acc)

        for i in range(len(args.target)):
            preds_index = target_pred[:, i] > 0.5
            preds_acc = (target[:, i] == preds_index).mean()
            target_accracy[i].append(preds_acc)

        # backpropagation
        optimizer.zero_grad()
        task_loss.backward()
        optimizer.step()

        if batch_idx == 500:
            break
    return task_accracy, target_accracy, original_loss_logger.avg, adv_loss_logger.avg, unet_loss_logger.avg, autoencoded_images[0:36]


def finetuningtest(args, epoch, model, criterion, criterionL1, val_loader):
    model.eval()
    task_loss_logger = AverageMeter()
    adv_loss_logger = AverageMeter()
    l1_loss_logger = AverageMeter()

    task_accracy = []
    for i in range(len(args.task)):
        task_accracy.append([])

    target_accracy = []
    for i in range(len(args.target)):
        target_accracy.append([])

    for batch_idx, sample in enumerate(tqdm(val_loader, desc='Val %d' % epoch)):
        images, task, target = sample['imgs'].float().cuda(), sample['task_labels'].float().cuda(), sample['target_labels'].float().cuda()
        task_pred, target_pred, autoencoded_images = model(images)

        unet_loss = criterionL1(autoencoded_images, images)
        task_loss = F.cross_entropy(task_pred, task.max(1, keepdim=False)[1], reduction='elementwise_mean')
        adv_loss = criterion(target_pred, target)

        task_loss_logger.update(task_loss.item())
        adv_loss_logger.update(adv_loss.item())
        l1_loss_logger.update(unet_loss.item())

        task_pred, target_pred = task_pred.cpu().detach().numpy(), target_pred.cpu().detach().numpy()
        task, target = task.cpu().detach().numpy(), target.cpu().detach().numpy()

        for i in range(len(args.task)):
            preds_index = task_pred[:, i] > 0.5
            preds_acc = (task[:, i] == preds_index).mean()
            task_accracy[i].append(preds_acc)

        for i in range(len(args.target)):
            preds_index = target_pred[:, i] > 0.5
            preds_acc = (target[:, i] == preds_index).mean()
            target_accracy[i].append(preds_acc)
        if batch_idx == 500:
            break
    return task_accracy, target_accracy, task_loss_logger.avg, adv_loss_logger.avg, l1_loss_logger.avg, autoencoded_images[0:36]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
    parser.add_argument('--save_path', type=str, default='./checkpoints/multiunlearning/', help='path for saving checkpoints')

    parser.add_argument('--adv_on', action='store_true', default=True, help='start adv training')
    parser.add_argument('--adv_lambda', type=float, default=10.0, help='weight assigned to adv loss')
    parser.add_argument('--beta', type=float, default=1.0, help='autoencoder l1 loss weight')

    parser.add_argument('--autoencoder_finetune', action='store_true', default=True)
    parser.add_argument('--finetune', action='store_true')

    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=50)

    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--img_size', type=int, default=224, help='Size of input image')

    parser.add_argument('--model', type=str, default='customresnet', choices=('customresnet', 'customefficientnet', 'effnetv2s'))
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--group_name', type=str, default='Finetuning Process Multi Labels task part')

    parser.add_argument('--wandb', default=True, action='store_true', help='Use wandb for logging')
    parser.add_argument('--task', type=list, default=['Bald'], help='Attributes')
    parser.add_argument('--target', type=list, default=['Mouth_Slightly_Open', 'Pointy_Nose'], help='Attributes')

    parser.add_argument('--training', default=True, action='store_true', help='Use wandb for logging')
    parser.add_argument('--multi', default=True, action='store_true', help='Use wandb for logging')

    parser.add_argument('--pretrained_task_network_path', type=str, default='./checkpoints/original/')
    parser.add_argument('--pretrained_target_network_path', type=str, default='./checkpoints/original_multi/')

    args = parser.parse_args()

    args.pretrained_task_network_path = os.path.join(args.pretrained_task_network_path, '_'.join(args.task), '_'.join(args.task) + "_final.pth")
    args.pretrained_target_network_path = os.path.join(args.pretrained_target_network_path, '_'.join(args.target), '_'.join(args.target) + "_final.pth")

    args.run_name = '_'.join(args.task) + "_" + '_'.join(args.target)

    for k, v in sorted(vars(args).items()):
        cprint.info(k, '=', v)

    main(args)
