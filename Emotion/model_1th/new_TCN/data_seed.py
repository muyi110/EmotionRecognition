# -*- coding:UTF-8 -*-
import os
import math
import numpy as np
import scipy.io as scio

def get_samples_data(path, trial_list, windows=4, overlapping=3):
    '''
      windows: 划分的时间窗口长度
      overlapping: 时间窗口的重叠长度
    '''
    datas = [] 
    labels = []
    data = scio.loadmat(path+"data1.mat")
    label = scio.loadmat(path+"label.mat")["label"] # shape = (1, 15)
    step = windows - overlapping
    for i in trial_list:
        data_channel = data["de_LDS"+str(i)].transpose((1,0,2))
        data_channel = data_channel.reshape(data_channel.shape[0], -1) # shape = (samples, 310)
        data_label = label[0][i-1] + 1
        numbers_single_trial = int((data_channel.shape[0] - windows) / step + 1)
        for iterator in range(numbers_single_trial):
            datas.append(data_channel[iterator*step:iterator*step + windows,:])
            labels.append(data_label)
    print("Get sample data success!")
    print("Total sample number is: ", len(labels))
    print("label 0: {}  label 1: {}  label 2: {}.".format(np.sum(np.array(labels)==0), 
                                                                 np.sum(np.array(labels)==1), 
                                                                 np.sum(np.array(labels)==2)))
    return (datas, labels)

def index_generator(num_examples, batch_size, seed=0):
    '''此函数用于生成 batch 的索引'''
    np.random.seed(seed)
    permutation = list(np.random.permutation(num_examples))
    num_complete_minibatches = math.floor(num_examples/batch_size)
    for k in range(0, num_complete_minibatches):
        X_batch_index = permutation[k*batch_size:(k+1)*batch_size]
        y_batch_index = permutation[k*batch_size:(k+1)*batch_size]
        yield (X_batch_index, y_batch_index)
    if num_examples % batch_size != 0:
        X_batch_index = permutation[num_complete_minibatches*batch_size:num_examples]
        y_batch_index = permutation[num_complete_minibatches*batch_size:num_examples]
        yield (X_batch_index, y_batch_index)

def read_data(path="../new_RNN/", trial_list=list(range(1, 16)), windows=9, overlapping=8):
    # datas 和 labels 都是 list. datas 中的每一项是 shape=(32, 128*windows) 的数组
    datas, labels = get_samples_data(path, trial_list, windows, overlapping)
    return (datas, labels)
