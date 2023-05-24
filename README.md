# ComputerVision_Group6
Topic: Weakly_supervised_crowd_counting.
Code for the Exam project in the course computer vision at Aarhus University, Denmark. <br /> 
Subject: Weakly supervised crowd-counting methods. 

The names of the two models are TransCrowd and CCTrans.  

## Report
Link to Overleaf: [link]()

## Models/Papers
The tested models: <br />
+ TransCrowd: weakly-supervised crowd counting with transformers (26 April 2022) <br />
Github: https://github.com/dk-liang/TransCrowd <br />

+ CCTrans: Simplifying and Improving Crowd Counting with Transformer (not published 29 Sep 2021) <br />
Github: https://github.com/wfs123456/CCTrans (reproduction)<br />

## Datasets
+ ShanghaiTech Part A: (https://www.kaggle.com/datasets/tthien/shanghaitech)<br />
  Part A: <br />
  Number of images:       482 <br />
  Number of Annotations:  241,677 <br />
  Avage Count per image:  501 <br />

+ JHU-CROWD++ can be downloaded at: (http://www.crowd-counting.com/) <br />
  Number of images:       4372 <br />
  Number of Annotations:  1,51 million <br />
  Avage Count per image:  346 <br />

## Environment
	python >=3.6 
	pytorch >=1.5
	opencv-python >=4.0
	scipy >=1.4.0
	h5py >=2.10
	pillow >=7.0.0
	imageio >=1.18
	timm==0.1.30
# run code
## TransCrowd
Prepare:
- Download pretrained weights to the transformer-encoder: https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth<br />
- Change path in models.py line 110. <br />
- Change filepath in predataset_sh.py do the same for predataset_jhu.py
```
python /data/predataset_sh.py
```
### Train:
Update argument dataset, test_dataset, gpu_id in config.py  
For shanghaiTech Part A dataset:
```
python train_cross_val.py 
```
For JHU-CROWD++ dataset:
```
python train.py 
```
### Test:
```
python test.py
```
## CCTrans
Prepare:
- Download pretrained weights: //github.com/Meituan-AutoML/Twins/alt_gvt_large.pth <br />
- Change path in /Networks/ALTGVT.py line 573. <br />
Prepare data for JHU-CROWD++:
```
python predataset_jhu.py
```
### Train:
Update arguments in train.py and train_five_fold.py  
For shanghaiTech Part A dataset:
```
python train_five_fold.py 
```
For JHU-CROWD++ dataset:
```
python train.py 
```
### Test:
```
python test_image_patch.py
```
### For visualization of density maps:
Change the image_patch in vis_densityMap.py
```
python vis_densityMap.py
```
## Description of files:
### TransCrowd
/data/predataset_sh.py (precrop images to 256X256 pixels and save ground true as .h5 file) <br />
/Networks/models.py (the model)<br />
config.py (arguments)<br />
train.py (train code)<br />
test.py (test code)<br />
dataset.py (load data and do argumentation)<br />
make_npydata.py (make train, val and test list)<br />
train_cross_val.py (train with five-fold cross validation)<br />
image.py (dataloader)<br />


### CCTrans
predataset_jhu.py (precrop data to 512x512 pixels and save gt as .h5 file)<br />
train.py (Train model with early stopping)<br />
train_five_fold.py (Train model with k-fold cross validation)<br />
test_image_patch.py (Test model on test set)<br />
vis_densityMap.py (Generate gt density map and predicted density map)<br />
train_helper_ALTGVT.py (initialization of model and control of train and validation)<br />
/Networks/ALTCVT.py (The model)<br />
/datasets/crowd.py (dataloader and data augmentation)<br />
