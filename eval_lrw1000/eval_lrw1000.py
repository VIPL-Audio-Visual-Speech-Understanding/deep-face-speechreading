# coding: utf-8
import os, sys
import time

import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.classifier import C3D_ResNet
from word_dataset import *
from cvtransforms import *


def get_dataloader(args):
    dsets = {x: LRW1000FaceDataset(x, args.dataset, args.color_space, args.max_timesteps) 
for x in ['test', 'val']}
    dset_loaders = {x: DataLoader(dsets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=(x == 'train')) for x in ['test', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['test', 'val']}

    return dset_loaders, dset_sizes


def test(args, use_gpu):
    dset_loaders, dset_sizes = get_dataloader(args)

    model = C3D_ResNet(mode=args.mode, inputDim=512, hiddenDim=args.hidden_num, nLayers=args.rnn_layers, nClasses=1000, frameLen=args.max_timesteps, backend=args.backend, every_frame=args.every_frame, color_space=args.color_space, use_cbam=args.cbam)
    criterion = nn.CrossEntropyLoss()
    
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
        criterion = criterion.cuda()
    
    pretrained_dict = torch.load(args.ckpt_path)
    model = nn.DataParallel(model)
    model.load_state_dict(pretrained_dict['state_dict'], strict=False)
    model.eval()
    
    # Evaluation loop.
    running_loss, running_corrects, running_all = 0., 0., 0.
    phase = 'test'
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dset_loaders[phase]):
            inputs = inputs.float()
            if use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model(inputs)
            if args.every_frame:
                outputs = torch.mean(outputs, 1)
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, targets)
            
            cur_loss = loss.item()
            running_loss += cur_loss * inputs.size(0)
            running_corrects += torch.sum(preds == targets).item()
            running_all += inputs.size(0)
            running_acc = running_corrects / running_all
            
            if batch_idx == 0:
                since = time.time()
            elif batch_idx % 100 == 0 or batch_idx == len(dset_loaders[phase]) - 1:                        
                print ('Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tAcc: {:.4f}\tElapsed: {:5.0f}s\tETA: {:5.0f}s\r'.format(
                    running_all,
                    len(dset_loaders[phase].dataset),
                    100. * batch_idx / (len(dset_loaders[phase])-1),
                    running_loss / running_all,
                    running_acc,
                    time.time()-since,
                    (time.time()-since)*(len(dset_loaders[phase])-1) / batch_idx - (time.time()-since)))
                
    epoch_loss = running_loss / len(dset_loaders[phase].dataset)
    epoch_acc = running_corrects / len(dset_loaders[phase].dataset)
    print ('Test loss: {:.4f}\tAcc: {:.4f}'.format(epoch_loss, epoch_acc) + '\n')


def main():
    # Settings
    parser = argparse.ArgumentParser(description='LRW-1000 Face VSR Evaluation Code')
    parser.add_argument('--gpus', default='0', help='device to use')
    parser.add_argument('--ckpt_path', default='lrw1000_cutout_rgb_gru.pt', help='path to pretrained model')
    parser.add_argument('--dataset', default='/scratch/zhangyuanhang/LRW1000_v2', help='path to dataset')

    parser.add_argument('--mode', default='finetuneCE', help='backendCE, finetuneCE')
    parser.add_argument('--backend', default='gru', help='gru, tcn')
    parser.add_argument('--color_space', default='rgb', help='color space: rgb, gray')
    parser.add_argument('--every_frame', default=False, action='store_true', help='prediction based on every frame')
    parser.add_argument('--cbam', default=False, action='store_true', help='use CBAM in ResNet blocks')

    parser.add_argument('--max_timesteps', default=30, type=int, help='maximum number of timesteps')
    parser.add_argument('--hidden_num', default=512, type=int, help='number of hidden units in the RNN')
    parser.add_argument('--rnn_layers', default=2, type=int, help='number of hidden layers in the RNN')
    
    parser.add_argument('--batch_size', default=36, type=int, help='mini-batch size (default: 36)')
    parser.add_argument('--workers', default=16, type=int, help='number of data loader workers (default: 16)')
    
    args = parser.parse_args()
    print (vars(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    use_gpu = torch.cuda.is_available()
    test(args, use_gpu)


if __name__ == '__main__':

    main()
    # Usage:
    # python3 eval_lrw1000.py --gpus 2,3 --dataset /scratch/zhangyuanhang/LRW1000_v2 --mode finetuneCE --backend gru --batch_size 128
    