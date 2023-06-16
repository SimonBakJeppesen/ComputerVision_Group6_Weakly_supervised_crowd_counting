# _*_ coding: utf-8 _*_
# @author   : 王福森
# @time     : 2021/12/3 20:54
# @File     : vis_densityMap.py
# @Software : PyCharm
from Networks import ALTGVT
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as io
import torch
import os
import argparse
import torch.nn.functional as F
import cv2
import h5py
from scipy import fftpack

def vis(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')

    model_path = args.weight_path
    crop_size = args.crop_size
    image_path = args.image_path

    model = ALTGVT.alt_gvt_large(pretrained=True)
    model.to(device)
    model.load_state_dict(torch.load(model_path, device))
    model.eval()

    if not os.path.exists(image_path):
        print("not find image path!")
        exit(-1)
    
    dataset = "jhu"

    print("detect image '%s'..." % image_path)
    if not os.path.exists(image_path):
        print("not find image path!")
        exit(-1)

    if dataset == "jhu":
        '''
        mat = io.loadmat(
            image_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('.', '_ann.').replace(
                "UCF-QNRF-Nor", "UCF-QNRF"))
        points = mat["annPoints"]
        '''
        gt_path = image_path.replace('.jpg', '.txt').replace('images', 'gt')
        print(gt_path)
        mat = np.loadtxt(gt_path, delimiter=' ')
        points = mat 
        gt_count = len(mat)
    else:
        mat = io.loadmat(image_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace("IMG", "GT_IMG"))
        points = mat["image_info"][0, 0][0, 0][0]
        gt_count = len(points)
    
    image = Image.open(image_path).convert("RGB")
    wd, ht = image.size
    st_size = 1.0 * min(wd, ht)
    if st_size < crop_size:
        rr = 1.0 * crop_size / st_size
        wd = round(wd * rr)
        ht = round(ht * rr)
        st_size = 1.0 * min(wd, ht)
        image = image.resize((wd, ht), Image.BICUBIC)

    # image = np.asarray(image, dtype=np.float32)
    # if len(image.shape) == 2:  # expand grayscale image to three channel.
    #     image = image[:, :, np.newaxis]
    #     image = np.concatenate((image, image, image), 2)
    # vis_img = image.copy()

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    image = transform(image)
    
    Img_data = cv2.imread(image_path)
    
    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    
    if mat.ndim > 1:
        Gt_data = mat[:,:2]
    
        if Img_data.shape[1] >= Img_data.shape[0]:
            rate_1 = 1 #1536.0 / Img_data.shape[1]
            rate_2 = 1 #1024 / Img_data.shape[0]
            Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_2)
            Gt_data[:, 0] = Gt_data[:, 0] * rate_1
            Gt_data[:, 1] = Gt_data[:, 1] * rate_2

        elif Img_data.shape[0] > Img_data.shape[1]:
            rate_1 = 1 #1536.0 / Img_data.shape[0]
            rate_2 = 1 #1024.0 / Img_data.shape[1]
            Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_1)
            Gt_data[:, 0] = Gt_data[:, 0] * rate_2
            Gt_data[:, 1] = Gt_data[:, 1] * rate_1
        
        for count in range(0, len(Gt_data)):
            if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
                kpoint[int(Gt_data[count][1]), int(Gt_data[count][0])] = 1

    ## gausian kernal
    t = np.linspace(-10, 10, 30)
    bump = np.exp(-0.1*t**2)
    bump /= np.trapz(bump) # normalize the integral to 1

    # make a 2-D kernel out of it
    kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
    kernel_ft = fftpack.fft2(kernel, shape=kpoint.shape[:2], axes=(0, 1))

    # convolve
    img_ft = fftpack.fft2(kpoint, axes=(0, 1))
    # the 'newaxis' is to match to color direction
    img2_ft = kernel_ft[:, :] * img_ft
    img2 = fftpack.ifft2(img2_ft, axes=(0, 1)).real

    # clip values to range
    img2 = np.clip(img2, 0, 1)
    
    gt_dmap = img2
      
    
    with torch.no_grad():
        inputs = image.unsqueeze(0).to(device)
        crop_imgs, crop_masks = [], []
        b, c, h, w = inputs.size()
        rh, rw = args.crop_size, args.crop_size
        for i in range(0, h, rh):
            gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
            for j in range(0, w, rw):
                gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                mask = torch.zeros([b, 1, h, w]).to(device)
                mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                crop_masks.append(mask)
        crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))
        crop_preds = []
        nz, bz = crop_imgs.size(0), args.batch_size
        for i in range(0, nz, bz):
            gs, gt = i, min(nz, i + bz)
            crop_pred, _ = model(crop_imgs[gs:gt])

            _, _, h1, w1 = crop_pred.size()
            crop_pred = F.interpolate(crop_pred, size=(h1 * 8, w1 * 8), mode='bilinear', align_corners=True) / 64

            crop_preds.append(crop_pred)
        crop_preds = torch.cat(crop_preds, dim=0)

        # splice them to the original size
        idx = 0
        pred_map = torch.zeros([b, 1, h, w]).to(device)
        for i in range(0, h, rh):
            gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
            for j in range(0, w, rw):
                gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                idx += 1
        # for the overlapping area, compute average value
        mask = crop_masks.sum(dim=0).unsqueeze(0)
        pred_map = pred_map / mask
        pred_map = pred_map.squeeze(0).squeeze(0).cpu().data.numpy()
    return pred_map, gt_dmap, gt_count, Img_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument("--image_path", default='/home/cv06f23/Dataset/jhu_crowd_v2.0/jhu_crowd_v2.0/test/images/2168.jpg',
                        help="the image path to be detected.")
    parser.add_argument("--weight_path", default='/home/cv09f23/ComputerVision_Group6_Weakly_supervised_crowd_counting/CCTrans/ckpts/ALTGVT/CCTrans_input-512/best_model.pth',
                        help="the weight path to be loaded")
    parser.add_argument('--crop_size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--batch-size', type=int, default=1, help='train batch size')

    args = parser.parse_args()
    print(args)

    pred_map, gt_dmap, gt_count, Img_data = vis(args)

    save_path = "vis/%s"%(args.image_path.split("/")[-4]+"/"+args.image_path.split("/")[-1][:-4])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("predmap count is %.2f, gt_dmap count is %.2f, gt count is %d"%(pred_map.sum(),gt_dmap.sum(),gt_count))

    vis_img = pred_map
    # normalize density map values from 0 to 1, then map it to 0-255.
    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)
    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    cv2.imwrite("%s/pred_map.png" % save_path, vis_img)
    cv2.imwrite("%s/Img.png" % save_path, Img_data)

    # plt.imsave("%s/pred_map.png" % save_path, pred_map)
    plt.imsave("%s/gt_dmap.png" % save_path, gt_dmap, cmap = 'jet')

    print("the visual result saved in %s"%save_path)