import copy
import os
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

from models.models import CustomResNetFor_Single, CustomResNetFor_Multi
from utils.scheduler import CosineAnnealingWarmUpRestart


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


def unzip_img_files(path):
    imgs_zip_file = os.path.join(path, 'img_align_celeba.zip')
    img_path = os.path.join(path, 'img_align_celeba')
    if os.path.exists(imgs_zip_file):
        with ZipFile(imgs_zip_file, 'r') as zip:
            if not os.path.isdir(img_path):
                print('Extracting Image Files..... Just wait for a moment!!')
                zip.extractall(path)
                print('Finished')
            else:
                print('Image Files already extracted')
    else:
        raise FileNotFoundError


def unzip_files(path, source_file, target_file):
    source_path = os.path.join(path, source_file)
    target_path = os.path.join(path, target_file)
    if os.path.exists(path):
        with ZipFile(source_path, 'r') as zip:
            if not os.path.isfile(target_path):
                print(f'Extracting {source_file}..... Just wait for a moment!!')
                zip.extractall(path)
                print('Finished')
            else:
                print(f'{target_file} already extracted')
    else:
        raise FileNotFoundError


def get_partition(data_path, mode='train'):
    partition = {
        'train': 0,
        'valid': 1,
        'test': 2
    }
    partition_info = pd.read_csv(os.path.join(data_path, 'list_eval_partition.csv'))

    return list(partition_info[partition_info['partition'] == partition[mode]].index)


def get_optimizer(cfg, model):
    if cfg.optimizer == 'adamw':
        return torch.optim.AdamW(params=model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'adam':
        return torch.optim.Adam(params=model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError


def get_single_classification_model(model_name):
    if model_name == 'customresnet':
        return CustomResNetFor_Single()
    else:
        raise ValueError


def get_multi_classification_model(model_name, out_number):
    if model_name == 'customresnet':
        return CustomResNetFor_Multi(out_number)
    else:
        raise ValueError


def get_scheduler(scheduler, optimizer):
    if scheduler == 'none':
        return None
    elif scheduler == 'cosinewarmup':
        return CosineAnnealingWarmUpRestart(optimizer=optimizer,
                                            T_0=4,
                                            T_mult=1,
                                            eta_max=2e-4,
                                            T_up=1,
                                            gamma=0.5)
    else:
        print("Wrong Scheduler!!")
        raise ValueError


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_answer(probs_list):
    pass

def make_grid(data, size=(6, 6)):
    rows, columns = size[0], size[1]
    data = data.detach().numpy()

    if (data <= 1).all():
        data = (data * 255).astype(np.uint8)

    data_row, data_columns = [], []
    for row in range(rows):
        data_columns.clear()
        for column in range(columns):
            images = data[row * columns + column].squeeze().transpose(1, 2, 0)
            data_columns.append(images)
        data_row.append(np.hstack(copy.deepcopy(data_columns)))
    data = np.vstack(data_row)
    return data

