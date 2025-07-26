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


def train_one_epoch(epoch,
                    device,
                    model,
                    loader,
                    optimizer):
    model.train()

    adv_preds = []
    adv_truth = []
    train_loss_list = list()

    pbar = tqdm(loader, total=loader.__len__(), position=0, leave=True)
    for sample in pbar:
        imgs, labels = sample['imgs'].float().to(device), sample['labels'].float().to(device)
        probs = model(imgs)
        loss = F.cross_entropy(probs, labels.max(1, keepdim=False)[1], reduction='mean')
        train_loss = loss.item()
        train_loss_list.append(train_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = np.argmax(F.softmax(probs, dim=1).cpu().detach().numpy(), axis=1)
        adv_preds += probs.tolist()
        adv_truth += labels.cpu().max(1, keepdim=False)[1].numpy().tolist()

        adv_acc = accuracy_score(adv_truth, adv_preds)
        desc = f"Train Epoch : {epoch + 1}, Loss : {np.mean(train_loss_list):.2f}"
        desc = desc + ", Accuracy" + f": {adv_acc:.2f}"
        pbar.set_description(desc)

    adv_acc = accuracy_score(adv_truth, adv_preds)
    return np.mean(train_loss_list), adv_acc


def valid_one_epoch(epoch,
                    device,
                    model,
                    loader):
    adv_preds = []
    adv_truth = []
    valid_loss_list = list()

    model.eval()

    pbar = tqdm(loader, total=loader.__len__(), position=0, leave=True)
    for sample in pbar:
        imgs, labels = sample['imgs'].float().to(device), sample['labels'].float().to(device)
        probs = model(imgs)

        adv_loss = F.cross_entropy(probs, labels.max(1, keepdim=False)[1], reduction='mean')

        probs = np.argmax(F.softmax(probs, dim=1).cpu().detach().numpy(), axis=1)
        adv_preds += probs.tolist()
        adv_truth += labels.cpu().max(1, keepdim=False)[1].numpy().tolist()

        valid_loss = adv_loss.item()
        valid_loss_list.append(valid_loss)

        adv_acc = accuracy_score(adv_truth, adv_preds)
        desc = f"Valid Epoch : {epoch + 1}, Loss : {np.mean(valid_loss_list):.2f}"
        desc = desc + ", Accuray" + f": {adv_acc:.2f}"
        pbar.set_description(desc)

    adv_acc = accuracy_score(adv_truth, adv_preds)
    return np.mean(valid_loss_list), adv_acc


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
                train_loss, train_accuracies = train_one_epoch(epoch=epo,
                                                               device=device,
                                                               model=model,
                                                               loader=train_loader,
                                                               optimizer=optimizer)
            logs = {"Train Loss": train_loss}
            logs["Train Accuracy"] = train_accuracies
            logs["Learning Rate"] = get_lr(optimizer)

            with torch.no_grad():
                valid_loss, valid_accuracies = valid_one_epoch(epoch=epo,
                                                               device=device,
                                                               model=model,
                                                               loader=valid_loader)

            logs["Valid Loss"] = valid_loss
            logs["Valid Accuracy"] = valid_accuracies

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

    else:
        pretrained_adversary_network_path = args.save_path + "final.pth"
        model = torch.load(pretrained_adversary_network_path)
        with torch.no_grad():
            valid_loss, valid_accuracies = valid_one_epoch(cfg=args,
                                                           epoch=0,
                                                           device=device,
                                                           model=model,
                                                           loader=valid_loader)
        print(valid_accuracies)


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
    parser.add_argument('--save_path', type=str, default='./checkpoints/original/', help='Dir path to save model weights')

    parser.add_argument('--model', type=str, default='customresnet', choices=('customresnet', 'customefficientnet', 'effnetv2s'))
    parser.add_argument('--epoch', type=int, default=10, help='Total Number of epochs')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-06, help='Learning Rate')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=('adam', 'adamw'))
    parser.add_argument('--weight_decay', type=float, default=1e-06)
    parser.add_argument('--img_size', type=int, default=224, help='Size of input image')
    parser.add_argument('--scheduler', type=str, default='cosinewarmup', choices=('none', 'cosinewarmup'))

    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--group_name', type=str, default='Single Classification Training Process')
    parser.add_argument('--wandb', default=True, action='store_true', help='Use wandb for logging')

    parser.add_argument('--multi', default=False, action='store_true', help='Use wandb for logging')
    parser.add_argument('--is_training_process', default=True, action='store_true', help='Use wandb for logging')

    # Bald Mouth_Slightly_Open
    parser.add_argument('--task_labels', type=list, default=['Mouth_Slightly_Open'], help='Attributes')

    args = parser.parse_args()
    args.run_name = '_'.join(args.task_labels)

    for k, v in sorted(vars(args).items()):
        cprint.info(k, '=', v)

    training(args)
