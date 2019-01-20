# -*- coding:UTF-8 -*-
from scipy.io import loadmat
import numpy as np
import sys

def dense_to_one_hot(labels_dense):
    labels_one_hot = np.zeros((len(labels_dense), ))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
            labels_one_hot[i] = t
        else:
            labels_one_hot[i] = t
    return labels_one_hot

def load_svhn():
    svhn_train = loadmat("../data/train_32×32.mat")
    svhn_test = loadmat("../data/test_32×32.mat")
    svhn_train_im = svhn_train['X']
    svhn_train_im = svhn_train_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label = dense_to_one_hot(svhn_train['y'])
    svhn_test_im = svhn_test['X']
    svhn_test_im = svhn_test_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label_test = dense_to_one_hot(svhn_test['y'])
    return svhn_train_im, svhn_label, svhn_test_im, svhn_label_test
