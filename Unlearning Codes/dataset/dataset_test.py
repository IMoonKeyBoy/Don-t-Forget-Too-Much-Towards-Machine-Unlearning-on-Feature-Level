import sys
from typing import Any
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from aif360.datasets import BinaryLabelDataset
from sklearn.preprocessing import StandardScaler


class ImageDataset(Dataset):
    """
    img_path: path to the image folder
    img_idxs: list of selected image indexes
    img_name_list: list of ALL image names, ['000001.jpg', '000002.jpg', ...]
    y_list: index of the label
    a_list: index of the attribute
    """

    def __init__(self, args, img_idxs, img_name_list, y_list, a_list, transform):
        self.img_path = args.img_path
        self.img_idxs = img_idxs
        self.img_names = img_name_list
        self.y_list = y_list
        self.a_list = a_list
        self.transform = transform

    def __len__(self):
        return len(self.img_idxs)

    def __getitem__(self, index1):
        img_id = self.img_idxs[index1]
        ta = self.y_list[index1]
        sa = self.a_list[index1]

        new_ta = np.zeros((2))
        if ta == 1:
            new_ta[0] = int(1)
            new_ta[1] = int(0)
        else:
            new_ta[0] = int(0)
            new_ta[1] = int(1)

        new_sa = np.zeros((2))
        if sa == 1:
            new_sa[0] = int(1)
            new_sa[1] = int(0)
        else:
            new_sa[0] = int(0)
            new_sa[1] = int(1)

        img1 = Image.open(self.img_path + self.img_names[img_id])
        sample = {'imgs': self.transform(img1), 'task_labels': new_ta, "target_labels": new_sa}
        return sample


def _prepare_data(args, a_list, y_list, img_names, transform, trial=0):
    index_11 = np.where((a_list == 1) & (y_list == 1))[0]
    index_10 = np.where((a_list == 1) & (y_list == 0))[0]
    index_01 = np.where((a_list == 0) & (y_list == 1))[0]
    index_00 = np.where((a_list == 0) & (y_list == 0))[0]
    a = [len(index_11), len(index_10), len(index_01), len(index_00)]
    print(a)
    # UTKFace [6713, 6914, 4601, 5477]
    # FairFace [6096, 6137, 8701, 7826]
    # FairFace train + val [6895, 6894, 9823, 8789]
    # CelebA ==============================
    # ya = [96759, 59975, 7074, 38791]
    # yg = [53447, 103287, 30987, 14878]
    num_minority = args.num_minority
    biase_ratio = args.biase_ratio
    num_majority = num_minority * biase_ratio
    np.random.shuffle(index_00)
    biased_train_index_00, test_index_00, remain_index_00 = np.split(index_00, [num_minority, num_minority + args.test_size])

    np.random.shuffle(index_01)
    biased_train_index_01, test_index_01, remain_index_01 = np.split(index_01, [num_majority, num_majority + args.test_size])

    np.random.shuffle(index_10)
    biased_train_index_10, test_index_10, remain_index_10 = np.split(index_10, [num_majority, num_majority + args.test_size])

    np.random.shuffle(index_11)
    biased_train_index_11, test_index_11, remain_index_11 = np.split(index_11, [num_minority, num_minority + args.test_size])

    biased_train_idxs = np.concatenate((biased_train_index_00, biased_train_index_01, biased_train_index_10, biased_train_index_11))

    balanced_test_idxs = np.concatenate((test_index_00, test_index_01, test_index_10, test_index_11))

    biased_dataset = ImageDataset(args,
                                  biased_train_idxs,
                                  img_names,
                                  y_list[biased_train_idxs],
                                  a_list[biased_train_idxs],
                                  transform)
    balanced_test_dataset = ImageDataset(args,
                                         balanced_test_idxs,
                                         img_names,
                                         y_list[balanced_test_idxs],
                                         a_list[balanced_test_idxs],
                                         transform)

    train_size = int(0.8 * len(biased_dataset))
    test_size = len(biased_dataset) - train_size
    biased_train_dataset, biased_val_dataset = torch.utils.data.random_split(biased_dataset, [train_size, test_size])

    biased_train_loader = torch.utils.data.DataLoader(biased_train_dataset,
                                                      batch_size=args.batch_size,
                                                      drop_last=True,
                                                      num_workers=6,
                                                      shuffle=True)
    biased_val_loader = torch.utils.data.DataLoader(biased_val_dataset,
                                                    batch_size=args.batch_size,
                                                    drop_last=True,
                                                    num_workers=6,
                                                    shuffle=True)
    balanced_test_loader = torch.utils.data.DataLoader(balanced_test_dataset,
                                                       batch_size=args.batch_size,
                                                       drop_last=True,
                                                       num_workers=6,
                                                       shuffle=False)

    '''
    loader_collect = {'biased_train_loader': biased_train_loader,
                    'biased_val_loader': biased_val_loader,
                    'balanced_test_loader': balanced_test_loader,
                    'remain_index': remain_index}

    torch.save(loader_collect, args.save_path + f'./loader_collect_{trial}.pt')
    '''
    return biased_train_loader, biased_val_loader, balanced_test_loader


def prepare_celeba(args, trial=0):
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    df = pd.read_csv(args.index_path + 'list_attr_celeba.txt', delim_whitespace=True, header=1)
    df.replace(-1, 0, inplace=True)
    img_names = df.index.tolist()
    a_list = df.values[:, args.a_index]
    y_list = df.values[:, args.y_index]
    return _prepare_data(args, a_list, y_list, img_names, transform, trial)
