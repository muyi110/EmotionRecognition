# -*- coding:UTF-8 -*-
import sys
from mnist import load_mnist
from svhn import load_svhn
from usps import load_usps

def return_dataset(data):
    if data == "svhn":
        train_image, train_label, test_image, test_label = load_svhn()
    if data == "mnist":
        train_image, train_label, test_image, test_label = load_mnist()
    if data == "usps":
        train_image, train_label, test_image, test_label = load_usps()

    return train_image, train_label, test_image, test_label

def dataset_read(source, target, batch_size):
    S = {}
    S_test = {}
    T = {}
    T_test = {}
    train_source, s_label_train, test_source, s_label_test = return_dataset(source)
    train_target, t_label_train, test_target, t_label_test = return_dataset(target)
    S['imgs'] = train_source
    S['labels'] = s_label_train
    T['imgs'] = train_target
    T['labels'] = t_label_train
    
    S_test['imgs'] = test_source
    S_test['labels'] = s_label_test
    T_test['imgs'] = test_target
    T_test['labels'] = t_label_test
    
    return S, S_test, T, T_test
