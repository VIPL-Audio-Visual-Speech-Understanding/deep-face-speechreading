# encoding: utf-8
import numpy as np
from tqdm import tqdm

import glob, os
import time
import pickle

import random
from cvtransforms import *
from torch.utils.data import Dataset


w, h = 122, 122
th, tw = 112, 112

x1_c = int(round((w - tw))/2.)
y1_c = int(round((h - th))/2.)


# face crops
def load_faces(imgpath, st, ed, color_space='rgb', is_training=False, max_len=30, cutout=True):
    imgs = []
    border_x = x1_c
    border_y = y1_c
    
    x1 = random.randint(0, border_x * 2) if is_training else border_x
    y1 = random.randint(0, border_y * 2) if is_training else border_y
    flip = is_training and random.random() > 0.5

    if ed > st + max_len: ed = st + max_len
    if st == ed: ed = st + 1 # safeguard
    files = [os.path.join(imgpath, '{:05d}.jpg'.format(i)) for i in range(st, ed)]
    files = list(filter(lambda path: os.path.exists(path), files))
#     assert len(files) > 0, 'exception: {}, {}, {}'.format(imgpath, st, ed)
    
    for frameCnt, fp in enumerate(files):
        try:
            img = cv2.imread(fp)
            rsz = img[y1: y1 + th, x1: x1 + tw]
            if flip: rsz = cv2.flip(rsz, 1)
            if color_space == 'gray': rsz = cv2.cvtColor(rsz, cv2.COLOR_BGR2GRAY)
            rsz = rsz / 255.
            if color_space == 'gray': rsz = ColorNormalize(rsz)
            imgs.append(rsz)
        except Exception as e:
            print (e, file)

    if len(imgs) == 0:
        seq = np.zeros((3, max_len, th, tw))
    else:
        seq = np.stack(imgs).transpose(3, 0, 1, 2) # RGB: THWC->CTHW
    # pad to max number of timesteps
    if seq.shape[1] < max_len:
        to_pad = max_len - seq.shape[1]
        seq = np.pad(seq, ((0, 0), (0, to_pad), (0, 0), (0, 0)), 'constant')
    if cutout and is_training:
        seq = SeqCutout(seq)

    return seq.astype(np.float32)


class LRW1000FaceDataset(Dataset):
    def __init__(self, folds, path='/scratch/zhangyuanhang/LRW1000_v2', color_space='rgb', max_len=30):
        self.folds = folds
        self.path = path
        self.color_space = color_space
        self.max_len = max_len

        dset = {'train': 'trn', 'test': 'tst', 'val': 'val'}
        index_root = os.path.join(self.path, 'info', '{}_1000.txt'.format(dset[self.folds]))
        lines = open(index_root, 'r').read().splitlines()
        lines = [line.split(',') for line in lines]
        pinyins = sorted([l.split('\t')[0] for l in open('lrw1000_words.txt', 'r').read().splitlines()])
        self.labels = pinyins

        # [file_name, start_frame, end_frame, class_idx]
        data_files = [(line[0], int(float(line[3]) * 25), int(float(line[4]) * 25), pinyins.index(line[2])) for line in lines]

        # filter training samples
        if self.folds == 'train':
            cache_path = 'train_1000.cache'
            try:
                self.data_files = pickle.load(open(cache_path, 'rb'))
            except:
                print ('Train: creating cache...')
                self.data_files = []
                print ('Data integrity verification...')
                for sample in tqdm(data_files):
                    path, st, ed, _ = sample
                    files = [os.path.join(self.path, 'images', path, '{:05d}.jpg'.format(i)) for i in range(st, ed)]
                    files = list(filter(lambda path: os.path.exists(path), files))
                    if len(files) > 0: self.data_files.append(sample)
                self.data_files = list(filter(lambda data: data[2] - data[1] <= self.max_len, self.data_files))
                print ('Retaining {} of {} samples.'.format(len(self.data_files), len(data_files)))
                pickle.dump(self.data_files, open(cache_path, 'wb'))
        else:
            self.data_files = data_files
        print ('Loaded {} set'.format(self.folds))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        path, st, ed, label = self.data_files[idx]
        inputs = load_faces(os.path.join(self.path, 'images', path), st, ed, self.color_space, self.folds == 'train', self.max_len)

        return inputs, label
