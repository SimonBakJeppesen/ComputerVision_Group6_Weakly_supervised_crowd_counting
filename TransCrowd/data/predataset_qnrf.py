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

'''change your dataset'''
root = '/home/dkliang/projects/synchronous/dataset/UCF-QNRF_ECCV18/'

img_train_path = os.path.join(root, 'Test/', 'images')
img_test_path = os.path.join(root, 'Train/', 'images')


path_sets = [img_train_path, img_test_path]

'''create path'''
if not os.path.exists(img_train_path.replace('images', 'gt_density_map_crop')):
    os.makedirs(img_train_path.replace('images', 'gt_density_map_crop'))

if not os.path.exists(img_test_path.replace('images', 'gt_density_map_crop')):
    os.makedirs(img_test_path.replace('images', 'gt_density_map_crop'))

if not os.path.exists(img_train_path.replace('images', 'images_crop')):
    os.makedirs(img_train_path.replace('images', 'images_crop'))

if not os.path.exists(img_test_path.replace('images', 'images_crop')):
    os.makedirs(img_test_path.replace('images', 'images_crop'))

















img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths.sort()

np.random.seed(0)
random.seed(0)
for img_path in img_paths:

    Img_data = cv2.imread(img_path)
    
    Gt_data = scipy.io.loadmat(img_path.replace('.jpg', '_ann.mat'))
    Gt_data = Gt_data['annPoints']

    rate = 1
    rate_1 = 1
    rate_2 = 1
    flag = 0
    if Img_data.shape[1] >= Img_data.shape[0]:
        rate_1 = 1152.0 / Img_data.shape[1]
        rate_2 = 768 / Img_data.shape[0]
        Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_2)
        Gt_data[:, 0] = Gt_data[:, 0] * rate_1
        Gt_data[:, 1] = Gt_data[:, 1] * rate_2

    elif Img_data.shape[0] > Img_data.shape[1]:
        rate_1 = 1152.0 / Img_data.shape[0]
        rate_2 = 768.0 / Img_data.shape[1]
        Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_1)
        Gt_data[:, 0] = Gt_data[:, 0] * rate_2
        Gt_data[:, 1] = Gt_data[:, 1] * rate_1
        print(img_path)


    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    
    for count in range(0, len(Gt_data)):
        if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
            kpoint[int(Gt_data[count][1]), int(Gt_data[count][0])] = 1

    height, width = Img_data.shape[0], Img_data.shape[1]

    m = int(width / 384)
    n = int(height / 384)
    fname = img_path.split('/')[-1]
    root_path = img_path.split('IMG_')[0].replace('images', 'images_crop')

    kpoint = kpoint.copy()
    if root_path.split('/')[-2] == 'Train':

        for i in range(0, m):
            for j in range(0, n):
                crop_img = Img_data[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384, ]
                crop_kpoint = kpoint[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384, ]
                gt_count = np.sum(crop_kpoint)

                save_fname = str(i) + str(j) + str('_') + fname
                save_path = root_path + save_fname
                
                h5_path = save_path.replace('.jpg', '.h5').replace('images', 'gt_density_map')

                with h5py.File(h5_path, 'w') as hf:
                    hf['gt_count'] = gt_count
                
                cv2.imwrite(save_path, crop_img)

    else:
        save_path = root_path.replace('images', 'images_crop') + fname
        
        cv2.imwrite(save_path, Img_data)

        gt_count = np.sum(kpoint)
        with h5py.File(save_path.replace('.jpg', '.h5').replace('images', 'gt_density_map'), 'w') as hf:
            hf['gt_count'] = gt_count
