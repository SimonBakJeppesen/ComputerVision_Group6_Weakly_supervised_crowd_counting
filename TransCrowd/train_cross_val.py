from __future__ import division
import warnings
from Networks.models import base_patch16_384_token, base_patch16_384_gap
import torch.nn as nn
from torchvision import transforms
import dataset
import math
from utils import save_checkpoint, setup_seed
import torch
import os
import logging
import nni
from nni.utils import merge_parameter
from config import return_args, args
import numpy as np
from image import load_data
from sklearn.model_selection import KFold
import traceback

warnings.filterwarnings('ignore')
import time

setup_seed(args.seed)

logger = logging.getLogger('mnist_AutoML')


def main(args):
    if args['dataset'] == 'ShanghaiA':
        train_file = './npydata/ShanghaiA_train.npy'
        test_file = './npydata/ShanghaiA_test.npy'
    elif args['dataset'] == 'ShanghaiB':
        train_file = './npydata/ShanghaiB_train.npy'
        test_file = './npydata/ShanghaiB_test.npy'
    elif args['dataset'] == 'UCF_QNRF':
        train_file = './npydata/UCF_QNRF_train.npy'
        test_file = './npydata/UCF_QNRF_test.npy'
    elif args['dataset'] == 'JHU':
        train_file = './npydata/jhu_train.npy'
        test_file = './npydata/jhu_val.npy'
    elif args['dataset'] == 'NWPU':
        train_file = './npydata/nwpu_train.npy'
        test_file = './npydata/nwpu_val.npy'

    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile).tolist()

    print(len(train_list))
    
    train_data = pre_data(train_list, args, train=True)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=45) #same 'random' split each time, so continuable if crash
    init_lr = args['lr']
    init_best_pred = args['best_pred']

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_id']
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_data)):
        fold += 1
        if fold in [1,2,3,4]:
            continue
        
        print('Beginning {} fold'.format(fold))
        args['lr'] = init_lr
        args['best_pred'] = init_best_pred
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            args=args),
        batch_size=args['batch_size'], num_workers=args['workers'], sampler=train_subsampler)
        
        val_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, num_workers=args['workers'], train=False),
        batch_size=1, sampler=val_subsampler)
        
        if args['model_type'] == 'token':
            model = base_patch16_384_token(pretrained=True)
        else:
            model = base_patch16_384_gap(pretrained=True)

        model = nn.DataParallel(model, device_ids=[0])
        model = model.cuda()

        criterion = nn.L1Loss(size_average=False).cuda()

        optimizer = torch.optim.Adam(
            [  #
                {'params': model.parameters(), 'lr': args['lr']},
            ], lr=args['lr'], weight_decay=args['weight_decay'])
        
        #After epoch 300 lr is timed by gamma
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300], gamma=0.1, last_epoch=-1) 

        # args['save_path'] = args['save_path'] + str(args['rdt'])
        print(args['save_path'])
        if not os.path.exists(args['save_path']):
            os.makedirs(args['save_path'])

        if args['pre']:
            if os.path.isfile(args['pre']):
                print("=> loading checkpoint '{}'".format(args['pre']))
                checkpoint = torch.load(args['pre'])
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                args['start_epoch'] = checkpoint['epoch']
                args['best_pred'] = checkpoint['best_prec1']
            else:
                print("=> no checkpoint found at '{}'".format(args['pre']))

        torch.set_num_threads(args['workers'])

        for epoch in range(args['start_epoch'], args['epochs']):
            epoch += 1

            start = time.time()
            train(train_loader, model, criterion, optimizer, epoch, args, scheduler)
            end1 = time.time()

            if epoch % 4 == 0 and epoch >= 0:
                prec1 = validate(val_loader, model, args)
                end2 = time.time()
                is_best = prec1 < args['best_pred']
                args['best_pred'] = min(prec1, args['best_pred'])

                print(' * best MAE {mae:.3f} '.format(mae=args['best_pred']), args['save_path'], end1 - start, end2 - end1)
                with open(args['save_path']+'/fold{}.txt'.format(fold), 'a') as f:
                    f.write('Epoch {}, MAE {}, BestMAE {}\n'.format(epoch, prec1, args['best_pred']))
                
                try:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args['pre'],
                        'state_dict': model.state_dict(),
                        'best_prec1': args['best_pred'],
                        'optimizer': optimizer.state_dict(),
                    }, is_best, args['save_path'], fold='fold{}_'.format(fold))
                except:
                    print('Warning: No checkpoint was saved')
                



def pre_data(train_list, args, train):
    print("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img, gt_count = load_data(Img_path, args, train)

        blob = {}
        blob['img'] = img
        blob['gt_count'] = gt_count
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1

        '''for debug'''
        # if j> 10:
        #     break
    return data_keys


def train(train_loader, model, criterion, optimizer, epoch, args, scheduler):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    
    args['lr'] = optimizer.param_groups[0]['lr']
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, len(train_loader.dataset), args['lr']))

    model.train()
    end = time.time()

    for j, (fname, img, gt_count) in enumerate(train_loader):
        data_time.update(time.time() - end)
        img = img.cuda()

        out1 = model(img)
        gt_count = gt_count.type(torch.FloatTensor).cuda().unsqueeze(1)

        # print(out1.shape, kpoint.shape)
        loss = criterion(out1, gt_count)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if j % args['print_freq'] == 0:
            print('4_Epoch: [{}][{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, j, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    scheduler.step()


def validate(val_loader, model, args):
    print('begin test')
    batch_size = 1

    model.eval()

    mae = 0.0
    mse = 0.0
    visi = []
    index = 0

    for i, (fname, img, gt_count) in enumerate(val_loader):

        img = img.cuda()
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            out1 = model(img)
            count = torch.sum(out1).item()

        gt_count = torch.sum(gt_count).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)

        if i % 15 == 0:
            print('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))

    mae = mae * 1.0 / (len(val_loader) * batch_size)
    mse = math.sqrt(mse / (len(val_loader)) * batch_size)

    nni.report_intermediate_result(mae)
    print(' \n* MAE {mae:.3f}\n'.format(mae=mae), '* MSE {mse:.3f}'.format(mse=mse))

    return mae


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    print(params)

    try:
        main(params)
    except Exception:
        with open('error_message.txt', 'w') as f:
            traceback.print_exc(file=f)
