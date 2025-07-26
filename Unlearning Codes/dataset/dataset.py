import os
import sys
from glob import glob
import cv2
import numpy as np
import pandas as pd
import torch
from cprint import cprint
from torch.utils.data import Dataset
from tqdm import tqdm, trange


class CelebASingleClassifiation(Dataset):
    def __init__(self, data_path, task_labels=None, transforms=None, is_split_notask_and_task_dataset=True, is_training=None, is_task=None):
        super().__init__()
        self.root_path = data_path
        self.task = task_labels
        self.img_path = os.path.join(self.root_path, 'img_align_celeba')
        self.transforms = transforms
        self.preprocess_img_id()
        self.preprocess_label()
        if is_split_notask_and_task_dataset:
            self.split_notask_and_task_dataset(is_training, is_task)
        else:
            self.split_training_and_test_dataset(is_training)

    def preprocess_img_id(self):
        img_ids = list(glob(os.path.join(self.img_path, '*.jpg')))
        self.img_ids_sorted = sorted(img_ids, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    def preprocess_label(self):
        labels = pd.read_csv(os.path.join(self.root_path, 'list_attr_celeba.csv'))
        labels = labels[self.task]
        labels = labels.replace(-1, 0)
        temp = labels.loc[:, :].values
        new_temp = np.zeros((temp.shape[0], 2))
        for i in range(temp.shape[0]):
            if temp[i][0] == 1:
                new_temp[i][0] = int(1)
                new_temp[i][1] = int(0)
            if temp[i][0] == 0:
                new_temp[i][1] = int(1)
                new_temp[i][0] = int(0)
        self.final_task_labels = new_temp

    def __getitem__(self, index):
        imgs = cv2.imread(self.img_ids_sorted[index], cv2.IMREAD_COLOR)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB) / 255.
        if self.transforms:
            imgs = self.transforms(imgs)
        labels = torch.tensor(self.final_task_labels[index, :])
        sample = {'imgs': imgs, 'labels': labels}
        return sample

    def __len__(self):
        return len(self.img_ids_sorted)

    def split_training_and_test_dataset(self, is_training):
        train_dataset_length = len(self.img_ids_sorted) - int(len(self.img_ids_sorted) * 0.25)
        if is_training:
            self.final_task_labels = self.final_task_labels[0:train_dataset_length]
            self.img_ids_sorted = self.img_ids_sorted[0:train_dataset_length]
        else:
            self.final_task_labels = self.final_task_labels[train_dataset_length:]
            self.img_ids_sorted = self.img_ids_sorted[train_dataset_length:]

    def split_notask_and_task_dataset(self, is_training, is_task):

        self.temp_final_task_labels = np.zeros((0, self.final_task_labels.shape[1]))
        self.temp_final_img_ids_sorted = []

        train_dataset_length = len(self.img_ids_sorted) - int(len(self.img_ids_sorted) * 0.25)
        if is_training:
            perpare_task_labels_temp = self.final_task_labels[0:train_dataset_length]
            prepare_img_ids_sorted_temp = self.img_ids_sorted[0:train_dataset_length]

            if is_task:
                for i in trange(len(prepare_img_ids_sorted_temp)):
                    if perpare_task_labels_temp[i][1] == 0 and perpare_task_labels_temp[i][0] == 1:
                        self.temp_final_task_labels = np.vstack((self.temp_final_task_labels, perpare_task_labels_temp[i]))
                        self.temp_final_img_ids_sorted.append(prepare_img_ids_sorted_temp[i])
            else:
                for i in trange(len(prepare_img_ids_sorted_temp)):
                    if perpare_task_labels_temp[i][1] == 1 and perpare_task_labels_temp[i][0] == 0:
                        self.temp_final_task_labels = np.vstack((self.temp_final_task_labels, perpare_task_labels_temp[i]))
                        self.temp_final_img_ids_sorted.append(prepare_img_ids_sorted_temp[i])

        else:
            perpare_task_labels_temp = self.final_task_labels[train_dataset_length:]
            prepare_img_ids_sorted_temp = self.img_ids_sorted[train_dataset_length:]

            if is_task:
                for i in trange(len(prepare_img_ids_sorted_temp)):
                    if perpare_task_labels_temp[i][1] == 0 and perpare_task_labels_temp[i][0] == 1:
                        self.temp_final_task_labels = np.vstack((self.temp_final_task_labels, perpare_task_labels_temp[i]))
                        self.temp_final_img_ids_sorted.append(prepare_img_ids_sorted_temp[i])
            else:
                for i in trange(len(prepare_img_ids_sorted_temp)):
                    if perpare_task_labels_temp[i][1] == 1 and perpare_task_labels_temp[i][0] == 0:
                        self.temp_final_task_labels = np.vstack((self.temp_final_task_labels, perpare_task_labels_temp[i]))
                        self.temp_final_img_ids_sorted.append(prepare_img_ids_sorted_temp[i])

        self.final_task_labels = self.temp_final_task_labels
        self.img_ids_sorted = self.temp_final_img_ids_sorted


class ADV_CelebASingleClassifiation(Dataset):

    def __init__(self, data_path, task, target, transforms=None, is_split_notask_and_task_dataset=True, is_training=None, is_task=None):
        super().__init__()
        self.root_path = data_path  # root path of dataset
        self.task = task
        self.target = target
        self.img_path = os.path.join(self.root_path, 'img_align_celeba')
        self.transforms = transforms
        self.preprocess_img_id()
        self.preprocess_label()

        if is_split_notask_and_task_dataset:
            self.split_notask_and_task_dataset(is_training, is_task)
        else:
            self.split_training_and_test_dataset(is_training)

    def preprocess_img_id(self):
        img_ids = list(glob(os.path.join(self.img_path, '*.jpg')))
        self.img_ids_sorted = sorted(img_ids, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    def preprocess_label(self):
        task_labels = pd.read_csv(os.path.join(self.root_path, 'list_attr_celeba.csv'))
        task_labels = task_labels[self.task]
        task_labels = task_labels.replace(-1, 0)

        temp_task = task_labels.loc[:, :].values
        new_temp_task = np.zeros((temp_task.shape[0], 2))
        for i in range(temp_task.shape[0]):
            if temp_task[i][0] == 1:
                new_temp_task[i][0] = int(1)
                new_temp_task[i][1] = int(0)
            if temp_task[i][0] == 0:
                new_temp_task[i][1] = int(1)
                new_temp_task[i][0] = int(0)
        self.final_task_labels = new_temp_task

        target_labels = pd.read_csv(os.path.join(self.root_path, 'list_attr_celeba.csv'))
        target_labels = target_labels[self.target]
        target_labels = target_labels.replace(-1, 0)

        temp_target = target_labels.loc[:, :].values
        new_temp_adv = np.zeros((temp_target.shape[0], 2))
        for i in range(temp_target.shape[0]):
            if temp_target[i][0] == 1:
                new_temp_adv[i][0] = int(1)
                new_temp_adv[i][1] = int(0)
            if temp_target[i][0] == 0:
                new_temp_adv[i][1] = int(1)
                new_temp_adv[i][0] = int(0)
        self.final_target_labels = new_temp_adv

    def split_training_and_test_dataset(self, is_training):
        train_dataset_length = len(self.img_ids_sorted) - int(len(self.img_ids_sorted) * 0.25)
        if is_training:
            self.final_task_labels = self.final_task_labels[0:train_dataset_length]
            self.final_target_labels = self.final_target_labels[0:train_dataset_length]
            self.img_ids_sorted = self.img_ids_sorted[0:train_dataset_length]
        else:
            self.final_task_labels = self.final_task_labels[train_dataset_length:]
            self.final_target_labels = self.final_target_labels[train_dataset_length:]
            self.img_ids_sorted = self.img_ids_sorted[train_dataset_length:]

    def split_notask_and_task_dataset(self, is_training, is_task):

        self.temp_final_task_labels = np.zeros((0, self.final_task_labels.shape[1]))
        self.temp_final_target_labels = np.zeros((0, self.final_target_labels.shape[1]))
        self.temp_final_img_ids_sorted = []

        train_dataset_length = len(self.img_ids_sorted) - int(len(self.img_ids_sorted) * 0.25)

        if is_training:
            perpare_task_labels_temp = self.final_task_labels[0:train_dataset_length]
            prepare_target_labels_temp = self.final_target_labels[0:train_dataset_length]
            prepare_img_ids_sorted_temp = self.img_ids_sorted[0:train_dataset_length]

            if is_task:
                for i in trange(len(prepare_img_ids_sorted_temp)):
                    if prepare_target_labels_temp[i][1] == 0 and prepare_target_labels_temp[i][0] == 1:
                        self.temp_final_task_labels = np.vstack((self.temp_final_task_labels, perpare_task_labels_temp[i]))
                        self.temp_final_target_labels = np.vstack((self.temp_final_target_labels, prepare_target_labels_temp[i]))
                        self.temp_final_img_ids_sorted.append(prepare_img_ids_sorted_temp[i])
            else:
                for i in trange(len(prepare_img_ids_sorted_temp)):
                    if prepare_target_labels_temp[i][1] == 1 and prepare_target_labels_temp[i][0] == 0:
                        self.temp_final_task_labels = np.vstack((self.temp_final_task_labels, perpare_task_labels_temp[i]))
                        self.temp_final_target_labels = np.vstack((self.temp_final_target_labels, prepare_target_labels_temp[i]))
                        self.temp_final_img_ids_sorted.append(prepare_img_ids_sorted_temp[i])

        else:
            perpare_task_labels_temp = self.final_task_labels[train_dataset_length:]
            prepare_target_labels_temp = self.final_target_labels[train_dataset_length:]
            prepare_img_ids_sorted_temp = self.img_ids_sorted[train_dataset_length:]

            if is_task:
                for i in trange(len(prepare_img_ids_sorted_temp)):
                    if prepare_target_labels_temp[i][1] == 0 and prepare_target_labels_temp[i][0] == 1:
                        self.temp_final_task_labels = np.vstack((self.temp_final_task_labels, perpare_task_labels_temp[i]))
                        self.temp_final_target_labels = np.vstack((self.temp_final_target_labels, prepare_target_labels_temp[i]))
                        self.temp_final_img_ids_sorted.append(prepare_img_ids_sorted_temp[i])
            else:
                for i in trange(len(prepare_img_ids_sorted_temp)):
                    if prepare_target_labels_temp[i][1] == 1 and prepare_target_labels_temp[i][0] == 0:
                        self.temp_final_task_labels = np.vstack((self.temp_final_task_labels, perpare_task_labels_temp[i]))
                        self.temp_final_target_labels = np.vstack((self.temp_final_target_labels, prepare_target_labels_temp[i]))
                        self.temp_final_img_ids_sorted.append(prepare_img_ids_sorted_temp[i])

        self.final_task_labels = self.temp_final_task_labels
        self.final_target_labels = self.temp_final_target_labels
        self.img_ids_sorted = self.temp_final_img_ids_sorted

    def __getitem__(self, index):
        imgs = cv2.imread(self.img_ids_sorted[index], cv2.IMREAD_COLOR)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB) / 255.
        if self.transforms:
            imgs = self.transforms(imgs)

        originallabels = torch.tensor(self.final_task_labels[index, :])
        AdversaryLabels = torch.tensor(self.final_target_labels[index, :])

        sample = {'imgs': imgs, 'task_labels': originallabels, 'target_labels': AdversaryLabels}
        return sample

    def __len__(self):
        return len(self.img_ids_sorted)


class CelebAMultiClassifiation(Dataset):

    def __init__(self, data_path, task, transforms=None, is_training=None):
        super().__init__()
        self.root_path = data_path
        self.task = task
        self.img_path = os.path.join(self.root_path, 'img_align_celeba')
        self.transforms = transforms
        self.preprocess_img_id()
        self.preprocess_label()

        self.split_training_and_test_dataset(is_training)

    def preprocess_img_id(self):
        img_ids = list(glob(os.path.join(self.img_path, '*.jpg')))
        self.img_ids_sorted = sorted(img_ids, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    def preprocess_label(self):
        task_labels = pd.read_csv(os.path.join(self.root_path, 'list_attr_celeba.csv'))
        task_labels = task_labels[self.task]
        task_labels = task_labels.replace(-1, 0)
        self.final_task_labels = task_labels.loc[:, :].values

    def split_training_and_test_dataset(self, is_training):
        train_dataset_length = len(self.img_ids_sorted) - int(len(self.img_ids_sorted) * 0.25)
        if is_training:
            self.final_task_labels = self.final_task_labels[0:train_dataset_length]
            self.img_ids_sorted = self.img_ids_sorted[0:train_dataset_length]
        else:
            self.final_task_labels = self.final_task_labels[train_dataset_length:]
            self.img_ids_sorted = self.img_ids_sorted[train_dataset_length:]

    def __getitem__(self, index):
        imgs = cv2.imread(self.img_ids_sorted[index], cv2.IMREAD_COLOR)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB) / 255.
        if self.transforms:
            imgs = self.transforms(imgs)
        task_label = torch.tensor(self.final_task_labels[index, :])
        sample = {'imgs': imgs, 'labels': task_label}
        return sample

    def __len__(self):
        return len(self.img_ids_sorted)


class ADV_CelebAMultiClassifiation(Dataset):

    def __init__(self, data_path, task_labels, target_labels, transforms=None, is_split_notask_and_task_dataset=True, is_training=None, is_task=None):
        super().__init__()
        self.root_path = data_path
        self.task = task_labels
        self.target = target_labels
        self.img_path = os.path.join(self.root_path, 'img_align_celeba')
        self.transforms = transforms
        self.preprocess_img_id()
        self.preprocess_label()

        if is_split_notask_and_task_dataset:
            self.split_notask_and_task_dataset(is_training, is_task)
        else:
            self.split_training_and_test_dataset(is_training)

    def preprocess_img_id(self):
        img_ids = list(glob(os.path.join(self.img_path, '*.jpg')))
        self.img_ids_sorted = sorted(img_ids, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    def preprocess_label(self):
        task_labels = pd.read_csv(os.path.join(self.root_path, 'list_attr_celeba.csv'))
        task_labels = task_labels[self.task]
        task_labels = task_labels.replace(-1, 0)
        #
        temp = task_labels.loc[:, :].values
        new_temp = np.zeros((temp.shape[0], 2))
        for i in range(temp.shape[0]):
            if temp[i][0] == 1:
                new_temp[i][0] = int(1)
                new_temp[i][1] = int(0)
            if temp[i][0] == 0:
                new_temp[i][1] = int(1)
                new_temp[i][0] = int(0)
        self.final_task_labels = new_temp

        target_labels = pd.read_csv(os.path.join(self.root_path, 'list_attr_celeba.csv'))
        target_labels = target_labels[self.target]
        target_labels = target_labels.replace(-1, 0)
        self.final_target_labels = target_labels.loc[:, :].values

    def split_notask_and_task_dataset(self, is_training, is_task):

        self.temp_final_task_labels = np.zeros((0, self.final_task_labels.shape[1]))
        self.temp_final_target_labels = np.zeros((0, self.final_target_labels.shape[1]))
        self.temp_final_img_ids_sorted = []

        train_dataset_length = len(self.img_ids_sorted) - int(len(self.img_ids_sorted) * 0.25)
        if is_training:
            perpare_task_labels_temp = self.final_task_labels[0:train_dataset_length]
            perpare_target_labels_temp = self.final_target_labels[0:train_dataset_length]
            prepare_img_ids_sorted_temp = self.img_ids_sorted[0:train_dataset_length]
            if is_task:
                for i in trange(len(prepare_img_ids_sorted_temp)):
                    if sum(perpare_target_labels_temp[i]) == len(self.target):
                        self.temp_final_task_labels = np.vstack((self.temp_final_task_labels, perpare_task_labels_temp[i]))
                        self.temp_final_target_labels = np.vstack((self.temp_final_target_labels, perpare_target_labels_temp[i]))
                        self.temp_final_img_ids_sorted.append(prepare_img_ids_sorted_temp[i])
            else:
                for i in trange(len(prepare_img_ids_sorted_temp)):
                    if sum(perpare_target_labels_temp[i]) == 0:
                        self.temp_final_task_labels = np.vstack((self.temp_final_task_labels, perpare_task_labels_temp[i]))
                        self.temp_final_target_labels = np.vstack((self.temp_final_target_labels, perpare_target_labels_temp[i]))
                        self.temp_final_img_ids_sorted.append(prepare_img_ids_sorted_temp[i])

        else:
            perpare_task_labels_temp = self.final_task_labels[train_dataset_length:]
            perpare_target_labels_temp = self.final_target_labels[train_dataset_length:]
            prepare_img_ids_sorted_temp = self.img_ids_sorted[train_dataset_length:]
            if is_task:
                for i in trange(len(prepare_img_ids_sorted_temp)):
                    if sum(perpare_target_labels_temp[i]) == len(self.target):
                        self.temp_final_task_labels = np.vstack((self.temp_final_task_labels, perpare_task_labels_temp[i]))
                        self.temp_final_target_labels = np.vstack((self.temp_final_target_labels, perpare_target_labels_temp[i]))
                        self.temp_final_img_ids_sorted.append(prepare_img_ids_sorted_temp[i])
            else:
                for i in trange(len(prepare_img_ids_sorted_temp)):
                    if sum(perpare_target_labels_temp[i]) == 0:
                        self.temp_final_task_labels = np.vstack((self.temp_final_task_labels, perpare_task_labels_temp[i]))
                        self.temp_final_target_labels = np.vstack((self.temp_final_target_labels, perpare_target_labels_temp[i]))
                        self.temp_final_img_ids_sorted.append(prepare_img_ids_sorted_temp[i])

        self.final_task_labels = self.temp_final_task_labels
        self.final_target_labels = self.temp_final_target_labels
        self.img_ids_sorted = self.temp_final_img_ids_sorted

    def split_training_and_test_dataset(self, is_training):
        train_dataset_length = len(self.img_ids_sorted) - int(len(self.img_ids_sorted) * 0.25)
        if is_training:
            self.final_task_labels = self.final_task_labels[0:train_dataset_length]
            self.final_target_labels = self.final_target_labels[0:train_dataset_length]
            self.img_ids_sorted = self.img_ids_sorted[0:train_dataset_length]
        else:
            self.final_task_labels = self.final_task_labels[train_dataset_length:]
            self.final_target_labels = self.final_target_labels[train_dataset_length:]
            self.img_ids_sorted = self.img_ids_sorted[train_dataset_length:]

    def __getitem__(self, index):
        imgs = cv2.imread(self.img_ids_sorted[index], cv2.IMREAD_COLOR)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB) / 255.
        if self.transforms:
            imgs = self.transforms(imgs)
        task_label = torch.tensor(self.final_task_labels[index, :])
        target_label = torch.tensor(self.final_target_labels[index, :])

        sample = {'imgs': imgs, 'task_labels': task_label, 'target_labels': target_label}
        return sample

    def __len__(self):
        return len(self.img_ids_sorted)


class Fairness_CelebAMultiClassifiation(Dataset):

    def __init__(self, data_path, args, transforms=None, is_split_notask_and_task_dataset=True, is_training=None):
        super().__init__()
        self.root_path = data_path
        self.task = args.task_labels
        self.target = args.target_labels
        self.img_path = os.path.join(self.root_path, 'img_align_celeba')
        self.transforms = transforms
        self.preprocess_img_id()
        self.preprocess_label()

        if is_split_notask_and_task_dataset:
            self.split_notask_and_task_dataset(is_training)
        else:
            self.split_training_and_test_dataset(is_training)

    def preprocess_img_id(self):
        img_ids = list(glob(os.path.join(self.img_path, '*.jpg')))
        self.img_ids_sorted = sorted(img_ids, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    def preprocess_label(self):
        task_labels = pd.read_csv(os.path.join(self.root_path, 'list_attr_celeba.csv'))
        task_labels = task_labels[self.task]
        task_labels = task_labels.replace(-1, 0)
        #
        temp = task_labels.loc[:, :].values
        new_temp = np.zeros((temp.shape[0], 2))
        for i in range(temp.shape[0]):
            if temp[i][0] == 1:
                new_temp[i][0] = int(1)
                new_temp[i][1] = int(0)
            if temp[i][0] == 0:
                new_temp[i][1] = int(1)
                new_temp[i][0] = int(0)
        self.final_task_labels = new_temp

        target_labels = pd.read_csv(os.path.join(self.root_path, 'list_attr_celeba.csv'))
        target_labels = target_labels[self.target]
        target_labels = target_labels.replace(-1, 0)
        self.final_target_labels = target_labels.loc[:, :].values

    def split_notask_and_task_dataset(self, is_training):

        self.temp_final_task_labels = np.zeros((0, self.final_task_labels.shape[1]))
        self.temp_final_target_labels = np.zeros((0, self.final_target_labels.shape[1]))
        self.temp_final_img_ids_sorted = []

        train_dataset_length = len(self.img_ids_sorted) - int(len(self.img_ids_sorted) * 0.25)
        if is_training:
            perpare_task_labels_temp = self.final_task_labels[0:train_dataset_length]
            perpare_target_labels_temp = self.final_target_labels[0:train_dataset_length]
            prepare_img_ids_sorted_temp = self.img_ids_sorted[0:train_dataset_length]

            for i in trange(len(prepare_img_ids_sorted_temp)):
                if sum(perpare_target_labels_temp[i]) == len(self.target):
                    self.temp_final_task_labels = np.vstack((self.temp_final_task_labels, perpare_task_labels_temp[i]))
                    self.temp_final_target_labels = np.vstack((self.temp_final_target_labels, perpare_target_labels_temp[i]))
                    self.temp_final_img_ids_sorted.append(prepare_img_ids_sorted_temp[i])


        else:
            perpare_task_labels_temp = self.final_task_labels[train_dataset_length:]
            perpare_target_labels_temp = self.final_target_labels[train_dataset_length:]
            prepare_img_ids_sorted_temp = self.img_ids_sorted[train_dataset_length:]
            for i in trange(len(prepare_img_ids_sorted_temp)):
                if sum(perpare_target_labels_temp[i]) == len(self.target):
                    self.temp_final_task_labels = np.vstack((self.temp_final_task_labels, perpare_task_labels_temp[i]))
                    self.temp_final_target_labels = np.vstack((self.temp_final_target_labels, perpare_target_labels_temp[i]))
                    self.temp_final_img_ids_sorted.append(prepare_img_ids_sorted_temp[i])

        self.final_task_labels = self.temp_final_task_labels
        self.final_target_labels = self.temp_final_target_labels
        self.img_ids_sorted = self.temp_final_img_ids_sorted

    def split_training_and_test_dataset(self, is_training):
        train_dataset_length = len(self.img_ids_sorted) - int(len(self.img_ids_sorted) * 0.25)
        if is_training:
            self.final_task_labels = self.final_task_labels[0:train_dataset_length]
            self.final_target_labels = self.final_target_labels[0:train_dataset_length]
            self.img_ids_sorted = self.img_ids_sorted[0:train_dataset_length]
        else:
            self.final_task_labels = self.final_task_labels[train_dataset_length:]
            self.final_target_labels = self.final_target_labels[train_dataset_length:]
            self.img_ids_sorted = self.img_ids_sorted[train_dataset_length:]

    def __getitem__(self, index):
        imgs = cv2.imread(self.img_ids_sorted[index], cv2.IMREAD_COLOR)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB) / 255.
        if self.transforms:
            imgs = self.transforms(imgs)
        task_label = torch.tensor(self.final_task_labels[index, :])
        target_label = torch.tensor(self.final_target_labels[index, :])

        sample = {'imgs': imgs, 'task_labels': task_label, 'target_labels': target_label}
        return sample

    def __len__(self):
        return len(self.img_ids_sorted)
