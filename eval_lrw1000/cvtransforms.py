# coding: utf-8
import random
import cv2
import numpy as np
import torch


def ColorNormalize(img):
    mean = 0.413621
    std = 0.1700239
    img = (img - mean) / std

    return img


def SeqCutout(seq, n_holes=1):
    h = seq.shape[-2]
    w = seq.shape[-1]

    # length = h // 8
    length = 3 * h // 16

    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length, 0, h)
        y2 = np.clip(y + length, 0, h)
        x1 = np.clip(x - length, 0, w)
        x2 = np.clip(x + length, 0, w)

        seq[:, :, y1: y2, x1: x2] = 0.

    return seq
