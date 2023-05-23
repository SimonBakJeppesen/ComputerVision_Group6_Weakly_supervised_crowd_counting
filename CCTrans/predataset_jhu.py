import os
import time
import cv2
import h5py
import numpy as np
import scipy.io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter
import math
import torch

import glob
'''change your dataset'''
root = '/home/cv06f23/Dataset/jhu_crowd_v2.0/jhu_crowd_v2.0'
img_train_path = root + '/train/images/'
img_val_path = root + '/val/images/'
img_test_path = root + '/test/images/'


save_train_img_path = root + '/train_data_CC/images/'
save_val_img_path = root + '/val_data_CC/images/'
save_test_img_path = root + '/test_data_CC/images/'

img_train = []
img_val = []
img_test = []


path_sets = [img_train_path, img_val_path, img_test_path]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths.sort()


for img_path in img_paths:
    Img_data = cv2.imread(img_path)
    mat = np.loadtxt(img_path.replace('.jpg', '.txt').replace('images', 'gt'), delimiter=' ')
    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    
    if mat.ndim > 1:
        Gt_data = mat[:,:2]

        if Img_data.shape[1] >= Img_data.shape[0]:
            rate_1 = 1536.0 / Img_data.shape[1]
            rate_2 = 1024 / Img_data.shape[0]
            Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_2)
            Gt_data[:, 0] = Gt_data[:, 0] * rate_1
            Gt_data[:, 1] = Gt_data[:, 1] * rate_2

        elif Img_data.shape[0] > Img_data.shape[1]:
            rate_1 = 1536.0 / Img_data.shape[0]
            rate_2 = 1024.0 / Img_data.shape[1]
            Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_1)
            Gt_data[:, 0] = Gt_data[:, 0] * rate_2
            Gt_data[:, 1] = Gt_data[:, 1] * rate_1
        
        kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
        
        for count in range(0, len(Gt_data)):
            if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
                kpoint[int(Gt_data[count][1]), int(Gt_data[count][0])] = 1

    height, width = Img_data.shape[0], Img_data.shape[1]

    m = int(width / 512)
    n = int(height / 512)
    fname = img_path.split('/')[-1]
    root_path = img_path.split('images')[0]

    
    if root_path.split('/')[-2] == 'train':
        #print("Train:" )
        for i in range(0, m):
            for j in range(0, n):
                crop_img = Img_data[j * 512: 512 * (j + 1), i * 512:(i + 1) * 512, ]
                crop_kpoint = kpoint[j * 512: 512 * (j + 1), i * 512:(i + 1) * 512, ]
                gt_count = np.sum(crop_kpoint)

                #print("crop:")
                save_fname = str(i) + str(j) + str('_') + fname
                save_path = root_path.replace('train', 'train_data_CC/images') + save_fname
                h5_path = save_path.replace('.jpg', '.h5').replace('images', 'gt_density_map')

                with h5py.File(h5_path, 'w') as hf:
                    hf['gt_count'] = gt_count
                cv2.imwrite(save_path, crop_img)
                print(save_path)
                
    elif root_path.split('/')[-2] == 'val':
        save_path = root_path.replace('val', 'val_data_CC/images') + fname
        print(save_path)
        cv2.imwrite(save_path, Img_data)

        gt_count = np.sum(kpoint)

        with h5py.File(save_path.replace('.jpg', '.h5').replace('images', 'gt_density_map'), 'w') as hf:
            hf['gt_count'] = gt_count
            
    elif root_path.split('/')[-2] == 'test':
        save_path = root_path.replace('test', 'test_data_CC/images') + fname
        print(save_path)
        cv2.imwrite(save_path, Img_data)
        
        gt_count = np.sum(kpoint)
        
        with h5py.File(save_path.replace('.jpg', '.h5').replace('images', 'gt_density_map'), 'w') as hf:
            hf['gt_count'] = gt_count
            
    else:
        print('Something went wrong')