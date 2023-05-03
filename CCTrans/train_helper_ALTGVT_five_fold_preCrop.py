import os
import time
import sys
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Subset
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from datasets.crowd_five_fold import Crowd_qnrf, Crowd_nwpu, Crowd_sh, CustomDataset
from sklearn.model_selection import KFold ####

#from models import vgg19
from Networks import ALTGVT
from losses.ot_loss import OT_Loss
from utils.pytorch_utils import Save_Handle, AverageMeter
import utils.log_utils as log_utils
import wandb

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[
        1
    ]  # the number of points is not fixed, keep it as a list of tensor
    gt_discretes = torch.stack(transposed_batch[2], 0)
    return images, points, gt_discretes


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        args = self.args
        sub_dir = (
            "ALTGVT/{}-input-{}".format(
                args.run_name,
                args.crop_size,
                #args.wot,
                #args.wtv,
                #args.reg,
                #args.num_of_iter_in_ot,
                #args.norm_cood,
            )
        )

        self.save_dir = os.path.join("ckpts", sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        time_str = datetime.strftime(datetime.now(), "%m%d-%H%M%S")
        self.logger = log_utils.get_logger(
            os.path.join(self.save_dir, "train-{:s}.log".format(time_str))
        )
        log_utils.print_config(vars(args), self.logger)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            self.logger.info("using {} gpus".format(self.device_count))
        else:
            raise Exception("gpu is not available")

        downsample_ratio = 8
        if args.dataset.lower() == "qnrf":
            self.datasets = {
                x: Crowd_qnrf(
                    os.path.join(
                        args.data_dir, x), args.crop_size, downsample_ratio, x
                )
                for x in ["train", "val"]
            }
        elif args.dataset.lower() == "nwpu":
            self.datasets = {
                x: Crowd_nwpu(
                    os.path.join(
                        args.data_dir, x), args.crop_size, downsample_ratio, x
                )
                for x in ["train", "val"]
            }
        elif args.dataset.lower() == "sha" or args.dataset.lower() == "shb":
            self.datasets = {
                "train": Crowd_sh(
                    os.path.join(args.data_dir, "train_data"),
                    args.crop_size,
                    downsample_ratio,
                    "train",
                ),
                "val": Crowd_sh(
                    os.path.join(args.data_dir, "train_data"),
                    args.crop_size,
                    downsample_ratio,
                    "val",
                ),
            }
        elif args.dataset.lower() == "custom":
            self.datasets = {
                "train": CustomDataset(
                    args.data_dir, args.crop_size, downsample_ratio, method="train"
                ),
                "val": CustomDataset(
                    args.data_dir, args.crop_size, downsample_ratio, method="valid"
                ),
            }
        else:
            raise NotImplementedError
       

        self.start_epoch = 0
        
        # check if wandb has to log
        if args.wandb:
            self.wandb_run = wandb.init(
            config=args, project="CTTrans", name=args.run_name
        )
        else : 
            wandb.init(mode="disabled")
        
        if args.resume:
            self.logger.info("loading pretrained model from " + args.resume)
            suf = args.resume.rsplit(".", 1)[-1]
            if suf == "tar":
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"])
                self.start_epoch = checkpoint["epoch"] + 1
            elif suf == "pth":
                self.model.load_state_dict(
                    torch.load(args.resume, self.device))
        else:
            self.logger.info("random initialization")
        
        """
        self.ot_loss = OT_Loss(
            args.crop_size,
            downsample_ratio,
            args.norm_cood,
            self.device,
            args.num_of_iter_in_ot,
            args.reg,
        )
        """
        self.tv_loss = nn.L1Loss(reduction="none").to(self.device)          #
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)                              # 
        self.smoothL1 = nn.SmoothL1Loss(beta=self.args.beta).to(self.device)
        self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        self.best_mse = np.inf
        # self.best_count = 0

    def train(self):
        """training process"""
        args = self.args
        kfold = KFold(n_splits=5, shuffle=True, random_state=45)            # Kfold
        
        for self.fold, (self.train_part, self.val_part) in enumerate(kfold.split(self.datasets["train"])):
            self.fold += 1
            if self.fold in [6]:
                continue
            
            print(self.train_part)
            print(self.val_part)
            
            if self.fold > 0:
            
                self.start_epoch = 0

                time_str = datetime.strftime(datetime.now(), "%m%d-%H%M%S")
                self.logger = log_utils.get_logger(
                    os.path.join(self.save_dir, "train-{:s}-fold{}.log".format(time_str,self.fold))
                )

                self.model = ALTGVT.alt_gvt_large(pretrained=True)
                self.model.to(self.device)
                self.optimizer = optim.AdamW(
                    self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
                )

                #OBS!!!! Implement scheduler here
                #self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1000], gamma=0.3, last_epoch=-1)

                self.best_mae = np.inf
                self.best_mse = np.inf

                print('Beginning {} fold'.format(self.fold))

                args = self.args
                for epoch in range(self.start_epoch, args.max_epoch + 1):
                    self.logger.info(
                        "-" * 5 + "Epoch {}/{}".format(epoch, args.max_epoch) + "-" * 5
                    )
                    self.epoch = epoch
                    self.train_epoch()
                    if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                        self.val_epoch()

    def train_epoch(self):
        ###epoch_ot_loss = AverageMeter()
        epoch_ot_obj_value = AverageMeter()
        epoch_wd = AverageMeter()
        epoch_count_loss = AverageMeter()
        epoch_tv_loss = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode
        
        train_dataset = Subset(self.datasets["train"],self.train_part)
        val_dataset = Subset(self.datasets["val"],self.val_part)
            
        #print('train_dataset')
        #print(len(train_dataset))
        #print(len(val_dataset))
            
        self.dataloaderTrain = DataLoader(
            train_dataset,
            collate_fn=(default_collate),
            batch_size=(self.args.batch_size),
            shuffle=(True),
            num_workers=self.args.num_workers * self.device_count,
            pin_memory=(True),
        )
            
        self.dataloaderVal = DataLoader(
            val_dataset,
            collate_fn=(default_collate),
            batch_size=(1),
            shuffle=(False),
            num_workers=self.args.num_workers * self.device_count,
            pin_memory=(False),
        )

        for step, (inputs, points) in enumerate(self.dataloaderTrain):
            inputs = inputs.to(self.device)
            gd_count = np.array(points, dtype=np.float32)
            N = inputs.size(0)

            with torch.set_grad_enabled(True):
                outputs, outputs_normed = self.model(inputs)
                
                count_loss = self.smoothL1(                         # insert Smooth l1
                    outputs.sum(1).sum(1).sum(1),
                    torch.from_numpy(gd_count).float().to(self.device),
                )
                epoch_count_loss.update(count_loss.item(), N)
                loss = count_loss 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred_count = (
                    torch.sum(outputs.view(N, -1),
                              dim=1).detach().cpu().numpy()
                )
                pred_err = pred_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_mae.update(np.mean(abs(pred_err)), N)

                # log wandb
                wandb.log(
                    {
                        "train/TOTAL_loss": loss,
                        "train/count_loss": count_loss,
                        "train/pred_err": pred_err,
                    },
                    step=self.epoch,
                )
        
        #self.scheduler.step()
        self.logger.info(
            "Epoch {} Train, Loss: {:.2f}, Wass Distance: {:.2f}, "
            "Count Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec".format(
                self.epoch,
                epoch_loss.get_avg(),
                epoch_wd.get_avg(),
                epoch_count_loss.get_avg(),
                np.sqrt(epoch_mse.get_avg()),
                epoch_mae.get_avg(),
                time.time() - epoch_start,
            )
        )
        
        #save ckpt file
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(
            self.save_dir, "{}_ckpt.tar".format(self.epoch))
        
        '''      # Save model for every epoch
        torch.save(
            {
                "epoch": self.epoch,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_state_dict": model_state_dic,
            },
            save_path,
        )
        '''
        self.save_list.append(save_path)

    def val_epoch(self):
        print("start val")
        args = self.args
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        
        for inputs, count, name in self.dataloaderVal:
            with torch.no_grad():
                # nputs = cal_new_tensor(inputs, min_size=args.crop_size)
                inputs = inputs.to(self.device)
                #gd_count_val = np.array([len(p) for p in points], dtype=np.float32)    
                
                crop_imgs, crop_masks = [], []
                b, c, h, w = inputs.size()
                rh, rw = args.crop_size, args.crop_size
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                        mask = torch.zeros([b, 1, h, w]).to(self.device)
                        mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                        crop_masks.append(mask)
                crop_imgs, crop_masks = map(
                    lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks)
                )
                
                crop_preds = []
                nz, bz = crop_imgs.size(0), args.batch_size
                for i in range(0, nz, bz):
                    gs, gt = i, min(nz, i + bz)
                    crop_pred, _ = self.model(crop_imgs[gs:gt])

                    _, _, h1, w1 = crop_pred.size()
                    crop_pred = (
                        F.interpolate(
                            crop_pred,
                            size=(h1 * 8, w1 * 8),
                            mode="bilinear",
                            align_corners=True,
                        )
                        / 64
                    )

                    crop_preds.append(crop_pred)
                crop_preds = torch.cat(crop_preds, dim=0)
                  
                #outputs, outputs_normed = self.model(inputs)
                
                # splice them to the original size
                idx = 0
                pred_map = torch.zeros([b, 1, h, w]).to(self.device)
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                        idx += 1
                # for the overlapping area, compute average value
                
                mask = crop_masks.sum(dim=0).unsqueeze(0)
                outputs = pred_map / mask
                
                
                res = count[0].item() - torch.sum(outputs).item()      # TO See
                epoch_res.append(res)
               
             
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))

        self.logger.info(
            "Fold {} Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec".format(
                self.fold, self.epoch, mse, mae, time.time() - epoch_start
            )
        )

        # log wandb
        wandb.log({"val/MSE": mse, "val/MAE": mae}, step=self.epoch)

        model_state_dic = self.model.state_dict()
        # if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
        print("Comaprison", mae,  self.best_mae)
        if mae < self.best_mae:
            self.best_mse = mse
            self.best_mae = mae
            self.logger.info(
                "save best mse {:.2f} mae {:.2f} model epoch {}".format(
                    self.best_mse, self.best_mae, self.epoch
                )
            )
            print("Saving best model at {} epoch".format(self.epoch))
            model_path = os.path.join(
                self.save_dir, "best_model_fold{}.pth".format(self.fold)
            )
            torch.save(
                model_state_dic,
                model_path,
            )

            if args.wandb:
                artifact = wandb.Artifact("model", type="model")
                artifact.add_file(model_path)
                
                self.wandb_run.log_artifact(artifact)
            
            # torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
            # self.best_count += 1


def tensor_divideByfactor(img_tensor, factor=32):
    _, _, h, w = img_tensor.size()
    h, w = int(h // factor * factor), int(w // factor * factor)
    img_tensor = F.interpolate(
        img_tensor, (h, w), mode="bilinear", align_corners=True)

    return img_tensor


def cal_new_tensor(img_tensor, min_size=256):
    _, _, h, w = img_tensor.size()
    if min(h, w) < min_size:
        ratio_h, ratio_w = min_size / h, min_size / w
        if ratio_h >= ratio_w:
            img_tensor = F.interpolate(
                img_tensor,
                (min_size, int(min_size / h * w)),
                mode="bilinear",
                align_corners=True,
            )
        else:
            img_tensor = F.interpolate(
                img_tensor,
                (int(min_size / w * h), min_size),
                mode="bilinear",
                align_corners=True,
            )
    return img_tensor


if __name__ == "__main__":
    import torch

    print(torch.__file__)
    x = torch.ones(1, 3, 768, 1152)
    y = tensor_spilt(x)
    print(y.size())