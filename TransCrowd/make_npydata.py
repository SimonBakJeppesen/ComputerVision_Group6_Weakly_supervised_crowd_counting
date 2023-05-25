import os
import numpy as np

if not os.path.exists('./npydata'):
    os.makedirs('./npydata')

'''please set your dataset path'''
try:
    JHU_train_path = '/home/cv06f23/Dataset/jhu_crowd_v2.0/jhu_crowd_v2.0/train_data/images/'
    JHU_val_path = '/home/cv06f23/Dataset/jhu_crowd_v2.0/jhu_crowd_v2.0/val_data/images/'
    JHU_test_path = '/home/cv06f23/Dataset/jhu_crowd_v2.0/jhu_crowd_v2.0/test_data/images/'

    train_list = []
    for filename in os.listdir(JHU_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(JHU_train_path + filename)

    train_list.sort()
    np.save('./npydata/JHU_train.npy', train_list)
    
    val_list = []
    for filename in os.listdir(JHU_val_path):
        if filename.split('.')[1] == 'jpg':
            val_list.append(JHU_val_path + filename)

    val_list.sort()
    np.save('./npydata/JHU_val.npy', val_list)

    test_list = []
    for filename in os.listdir(JHU_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(JHU_test_path + filename)
    test_list.sort()
    np.save('./npydata/JHU_test.npy', test_list)

    print("generate JHU image list successfully", len(train_list), len(test_list))
except:
    print("The JHU dataset path is wrong. Please check you path.")

'''please set your dataset path'''
try:
    UCF_QNRF_train_path = '/home/cv06f23/Dataset/UCF-QNRF/UCF_QNRF_ECCV18/train_data/images/'
    UCF_QNRF_test_path = '/home/cv06f23/Dataset/UCF-QNRF/UCF_QNRF_ECCV18/test_data/images/'

    train_list = []
    for filename in os.listdir(UCF_QNRF_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(UCF_QNRF_train_path + filename)

    train_list.sort()
    np.save('./npydata/UCF_QNRF_train.npy', train_list)

    test_list = []
    for filename in os.listdir(UCF_QNRF_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(UCF_QNRF_test_path + filename)
    test_list.sort()
    np.save('./npydata/UCF_QNRF_test.npy', test_list)

    print("generate UCF_QNRF image list successfully", len(train_list), len(test_list))
except:
    print("The UCF_QNRF dataset path is wrong. Please check you path.")


try:
    shanghaiAtrain_path = '/home/cv06f23/Dataset/ShanghaiTech/ShanghaiTech/part_A/train_data/images_crop/'
    shanghaiAtest_path = '/home/cv06f23/Dataset/ShanghaiTech/ShanghaiTech/part_A/test_data/images_crop/'

    train_list = []
    for filename in os.listdir(shanghaiAtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiAtrain_path + filename)

    train_list.sort()
    np.save('./npydata/ShanghaiA_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiAtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiAtest_path + filename)
    test_list.sort()
    np.save('./npydata/ShanghaiA_test.npy', test_list)

    print("generate ShanghaiA image list successfully", len(train_list), len(test_list))
except:
    print("The ShanghaiA dataset path is wrong. Please check you path.")

try:
    shanghaiBtrain_path = '/home/cv06f23/Dataset/ShanghaiTech/ShanghaiTech/part_B/train_data/images_crop/'
    shanghaiBtest_path = '/home/cv06f23/Dataset/ShanghaiTech/ShanghaiTech/part_B/test_data/images_crop/'

    train_list = []
    for filename in os.listdir(shanghaiBtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiBtrain_path + filename)
    train_list.sort()
    np.save('./npydata/ShanghaiB_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiBtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiBtest_path + filename)
    test_list.sort()
    np.save('./npydata/ShanghaiB_test.npy', test_list)
    print("Generate ShanghaiB image list successfully", len(train_list), len(test_list))
except:
    print("The ShanghaiB dataset path is wrong. Please check your path.")
