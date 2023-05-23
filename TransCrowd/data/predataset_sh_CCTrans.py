import glob
import math
import os
import torch
import cv2
import h5py
import numpy as np
import scipy.io as io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter
import random

'''set your data path'''
root = '/home/cv06f23/Dataset/ShanghaiTech/ShanghaiTech/'

part_A_train = os.path.join(root, 'part_A/train_data', 'images')
part_A_test = os.path.join(root, 'part_A/test_data', 'images')


path_sets = [part_A_train, part_A_test]

'''for part A'''
if not os.path.exists(part_A_train.replace('images', 'gt_density_map_crop_CC')):
    os.makedirs(part_A_train.replace('images', 'gt_density_map_crop_CC'))

if not os.path.exists(part_A_test.replace('images', 'gt_density_map_crop_CC')):
    os.makedirs(part_A_test.replace('images', 'gt_density_map_crop_CC'))

if not os.path.exists(part_A_train.replace('images', 'images_crop_CC')):
    os.makedirs(part_A_train.replace('images', 'images_crop_CC'))

if not os.path.exists(part_A_test.replace('images', 'images_crop_CC')):
    os.makedirs(part_A_test.replace('images', 'images_crop_CC'))


img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths.sort()

np.random.seed(0)
random.seed(0)
for img_path in img_paths:

    Img_data = cv2.imread(img_path)

    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_'))
    Gt_data = mat["image_info"][0][0][0][0][0]

    rate = 1
    rate_1 = 1
    rate_2 = 1
    flag = 0
    if Img_data.shape[1] >= Img_data.shape[0]:  # 后面的大
        rate_1 = 768 / Img_data.shape[1]
        rate_2 = 512 / Img_data.shape[0]
        Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_2)
        Gt_data[:, 0] = Gt_data[:, 0] * rate_1
        Gt_data[:, 1] = Gt_data[:, 1] * rate_2

    elif Img_data.shape[0] > Img_data.shape[1]:  # 前面的大
        rate_1 = 768 / Img_data.shape[0]
        rate_2 = 512 / Img_data.shape[1]
        Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_1)
        Gt_data[:, 0] = Gt_data[:, 0] * rate_2
        Gt_data[:, 1] = Gt_data[:, 1] * rate_1
        print(img_path)

    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))

    for i in range(0, len(Gt_data)):
        if int(Gt_data[i][1]) < Img_data.shape[0] and int(Gt_data[i][0]) < Img_data.shape[1]:
            kpoint[int(Gt_data[i][1]), int(Gt_data[i][0])] = 1

    height, width = Img_data.shape[0], Img_data.shape[1]

    m = int(width / 256)
    n = int(height / 256)
    fname = img_path.split('/')[-1]
    root_path = img_path.split('IMG_')[0].replace('images', 'images_crop_CC')

    kpoint = kpoint.copy()
    if root_path.split('/')[-3] == 'train_data':

        for i in range(0, m):
            for j in range(0, n):
                crop_img = Img_data[j * 256: 256 * (j + 1), i * 256:(i + 1) * 256, ]
                crop_kpoint = kpoint[j * 256: 256 * (j + 1), i * 256:(i + 1) * 256]
                gt_count = np.sum(crop_kpoint)

                save_fname = str(i) + str(j) + str('_') + fname
                save_path = root_path + save_fname

                h5_path = save_path.replace('.jpg', '.h5').replace('images', 'gt_density_map')
                if gt_count == 0:
                    print(save_path, h5_path)
                with h5py.File(h5_path, 'w') as hf:
                    hf['gt_count'] = gt_count

                cv2.imwrite(save_path, crop_img)

    else:
        img_path = img_path.replace('images', 'images_crop_CC')

        cv2.imwrite(img_path, Img_data)

        gt_count = np.sum(kpoint)
        with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'gt_density_map'), 'w') as hf:
            hf['gt_count'] = gt_count
