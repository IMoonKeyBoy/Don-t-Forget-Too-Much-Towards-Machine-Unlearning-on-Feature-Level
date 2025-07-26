import argparse
import os
import shutil
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import trange, tqdm


def channel_max_min_whole(f_map):
    T, C, H, W = f_map.shape
    print(f_map.shape)
    max_v = np.max(f_map, axis=(0, 2, 3), keepdims=True)
    print(max_v.shape)
    min_v = np.min(f_map, axis=(0, 2, 3), keepdims=True)
    print(min_v.shape)

    return (f_map - min_v) / (max_v - min_v + 1e-6)


def draw_fmap_from_npz(data, save_dir, SHOW_NUM, save_channel):
    N, C, H, W = data.shape
    print('data shape:', data.shape)
    for i in range(N):
        if i in SHOW_NUM:
            for j in save_channel:  # range(10):
                fig = data[i, j]
                fig = cv.resize(fig, (112, 112))
                cv.imwrite(save_dir + 'sample' + str(i) + '_channel' + str(j) + '.bmp', fig * 255.0)


def addTransparency(img, factor=0.3):
    img = img.convert('RGBA')
    img_blender = Image.new('RGBA', img.size, (0, 0, 0, 0))
    img = Image.blend(img_blender, img, factor)
    return img


def put_mask(img_path, mask_path, output_fold, Th, factor):
    img = Image.open(img_path)
    img = addTransparency(img, factor)
    mask_img = cv.resize(cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR), (224, 224))
    ori_img = cv.resize(cv.imread(img_path), (224, 224))
    zeros_mask = cv.resize(cv.imread(mask_path), (224, 224))

    mask_for_red = np.zeros((224, 224))
    for i in range(zeros_mask.shape[0]):
        for j in range(zeros_mask.shape[1]):
            if np.sum((zeros_mask[i][j] / 255.0) > Th):  # vgg/cub 0.5 # VOC animal 0.5
                mask_for_red[i][j] = 1
                # mask_img[i][j] = 0#ori_img[i][j] here is mask image
                mask_img[i][j] = ori_img[i][j]
            else:
                mask_for_red[i][j] = 0
    # add blue range
    red = np.zeros((224, 224))
    for i in range(mask_for_red.shape[0]):
        for j in range(mask_for_red.shape[1]):
            if j > 2 and mask_for_red[i][j - 1] == 0 and mask_for_red[i][j] == 1:
                red[i][j] = 1
                red[i][j - 1] = 1
                red[i][j - 2] = 1
                red[i][j - 3] = 1
                if j < (mask_for_red.shape[1] - 2):
                    red[i][j + 1] = 1
                    red[i][j + 2] = 1
                    # red[i][j+3] = 1
            if j < (mask_for_red.shape[1] - 3) and mask_for_red[i][j] == 1 and mask_for_red[i][j + 1] == 0:
                red[i][j] = 1
                if j > 1:
                    red[i][j - 1] = 1
                    red[i][j - 2] = 1
                    # red[i][j-3] = 1
                red[i][j + 1] = 1
                red[i][j + 2] = 1
                red[i][j + 3] = 1
            if i > 2 and mask_for_red[i - 1][j] == 0 and mask_for_red[i][j] == 1:
                red[i - 1][j] = 1
                red[i - 2][j] = 1
                red[i - 3][j] = 1
                red[i][j] = 1
                if i < (mask_for_red.shape[0] - 2):
                    red[i + 1][j] = 1
                    red[i + 2][j] = 1
                    # red[i+3][j] = 1
            if i < (mask_for_red.shape[0] - 3) and mask_for_red[i][j] == 1 and mask_for_red[i + 1][j] == 0:
                if i > 1:
                    red[i - 1][j] = 1
                    red[i - 2][j] = 1
                    # red[i-3][j] = 1
                red[i][j] = 1
                red[i + 1][j] = 1
                red[i + 2][j] = 1
                red[i + 3][j] = 1
    for i in range(mask_for_red.shape[0]):
        for j in range(mask_for_red.shape[1]):
            if red[i][j] == 1:
                mask_img[i][j] = [255, 0, 0]

    return mask_img


# image add mask
def image_add_mask(show_num, image_dir, mask_dir, save_dir, save_channel, save_channel_title, factor, animal, show_num_per_center):
    images = []
    for i in show_num:
        if animal == 'bird':
            image_paths = image_dir + 'vocbird_' + str(i) + '.jpg'
        else:
            image_paths = image_dir + str(i) + '.jpg'
        for j, channel in enumerate(tqdm(save_channel)):
            mask_path = mask_dir + 'sample' + str(i) + '_channel' + str(channel) + '.bmp'
            mask_img = put_mask(img_path=image_paths, mask_path=mask_path, output_fold=save_dir, Th=Th, factor=factor)
            mask_img = cv.resize(mask_img, (112, 112))
            # cv.imwrite(os.path.join(save_dir+'factor'+str(factor)+'_Th'+str(Th)+'_sample'+str(i)+'_center'+str(j//show_num_per_center)+'_channel'+str(channel)+'.bmp'), mask_img)
            images.append(mask_img)

    columns = int(len(images) ** 0.5)
    rows = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(rows, columns))
    image_index = 1
    for i in trange(columns * rows):
        fig.add_subplot(rows, columns, image_index)
        image_index = image_index + 1
        plt.axis('off')
        plt.tight_layout()
        # plt.title(save_channel_title[i - 1])
        plt.imshow(images[i - 1], cmap="gray")
    plt.show()


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # add positional arguments

    parser.add_argument('-Th', type=int, default=0.2)
    parser.add_argument('-factor', type=int, default=0.5)
    parser.add_argument('-show_num', type=int, default=8)
    parser.add_argument('-model', default="vgg", type=str)
    parser.add_argument('-animal', default="celeba", type=str)
    parser.add_argument('-fmap_path', default="/home/hengxu/Data/PycharmWorkspace/Technical Paper 3/icCNN-main-new/icCNN/basic_fmap/vgg_iccnn_original_135/13_vgg_celeb_iccnn.npz", type=str)
    parser.add_argument('-loss_path', default="/home/hengxu/Data/PycharmWorkspace/Technical Paper 3/icCNN-main-new/icCNN/vgg/13_vgg_celeb_vgg_iccnn_135/loss_1200.npz", type=str)
    parser.add_argument('-folder_name', default=None, type=str)
    args = parser.parse_args()
    # fixed
    Th = args.Th  # >Th --> in the red circle
    factor = args.factor  # the smaller the factor, the darker the area outside the red circle
    animals = ['bird', 'cat', 'dog', 'cow', 'horse', 'sheep', 'cub', 'celeba']
    show_num_per_center = args.show_num
    file_path = args.fmap_path
    # the No. of sample to visualize; the No. starts from 0
    # The id of images that you want to visualize
    voc = [
        [3],  # voc_bird
        [1],  # voc_cat
        [1],  # voc_dog
        [1],  # voc_cow
        [1],  # voc_horse
        [1],  # voc_sheep
        [1],  # cub
        [1]  # celeba
    ]
    loss = np.load(args.loss_path)
    gt = loss['gt'][-1]  # show channel id of different groups
    print(gt.shape)
    cluster_label = get_cluster(gt)
    print('groups and channels', cluster_label)

    save_channel = []
    save_channel_title = []
    for i in range(len(cluster_label)):
        for j in range(len(cluster_label[i])):
            save_channel.append(cluster_label[i][j])
            # name = '_'.join(str(cluster_label[i]))
            save_channel_title.append(str(cluster_label[i][j]))

    print(len(save_channel))
    if args.folder_name == None:
        model_name = args.model + '_' + args.animal
    else:
        model_name = args.folder_name

    animal_id = animals.index(args.animal)  # 0~5; which category you want to draw feature maps
    SHOW_NUM = voc[animal_id]
    animal = animals[animal_id]

    # load data
    data = np.load(file_path)['f_map']
    data = channel_max_min_whole(data)  #
    save_dir = './fmap/' + model_name + '/' + animal + '/'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    else:
        os.makedirs(save_dir)
    # draw feature map and save feature maps
    draw_fmap_from_npz(data, save_dir=save_dir, SHOW_NUM=SHOW_NUM, save_channel=save_channel)  #############iccnn

    if args.animal == 'cub':
        img_dir = './images/hook_cub_test/'
    elif args.animal == 'celeba':
        img_dir = './images/hook_celeba_test/'
    else:
        img_dir = './images/voc' + animal + '_test/'
    mask_dir = './fmap/' + model_name + '/' + animal + '/'  # i.e. the dir of feature maps (same with the 'save_dir' above)
    masked_save_dir = './fmap/' + model_name + '/' + animal + '_masked/'  # save dir of images with the red circle we want!
    if os.path.exists(masked_save_dir):
        shutil.rmtree(masked_save_dir)
        os.makedirs(masked_save_dir)
    else:
        os.makedirs(masked_save_dir)
    image_add_mask(show_num=SHOW_NUM, image_dir=img_dir, mask_dir=mask_dir, save_dir=masked_save_dir, save_channel=save_channel, save_channel_title=save_channel_title, factor=factor, animal=animal,
                   show_num_per_center=show_num_per_center)
