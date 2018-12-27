# -*- coding:UTF-8 -*-
import os
import math
import numpy as np
import features
from features import data_filter, differential_entropy

SAMPLES_PATH = '../../samples/'
params = (features.b_theta, features.a_theta, features.b_alpha, features.a_alpha, 
          features.b_beta, features.a_beta, features.b_gamma, features.a_gamma)
def get_samples_data(path, people_list, trial_list, windows=4, overlapping=3):
    '''
      windows: 划分的时间窗口长度
      overlapping: 时间窗口的重叠长度
    '''
    samples_dirs = os.listdir(path) # 目录的顺序是随机的
    samples_dirs = sorted(samples_dirs)
    file_path = [os.path.join(path, samples_dirs[i]) for i in range(len(samples_dirs))]
    datas = [] 
    labels = []
    for people in people_list:
        for trial in trial_list:
            data = np.loadtxt(file_path[people]+'/trial_'+str(trial+1)+".csv", delimiter=',', 
                             skiprows=0, dtype=np.float32)
            data = data[:32,128*3:] # 只是提取后 60S 的 EEG 数据
            # 获取对应的 labels
            labels_value = np.loadtxt(file_path[people]+'/label.csv', delimiter=',', 
                                      skiprows=0, dtype=np.float32)[trial,:2]
            if labels_value[0] > 5:
                label = 1 # 第一象限
            elif labels_value[0] > 5.:
                label = 1 # 第二象限
            elif labels_value[0] <= 5.:
                label = 2 # 第三象限
            elif labels_value[0] <= 5.:
                label = 2 # 第四象限
            # if labels_value[0] > 5. and labels_value[1] > 5.:
            #     label = 1 # 第一象限
            # elif labels_value[0] >= 5. and labels_value[1] <= 5.:
            #     label = 1 # 第二象限
            # elif labels_value[0] < 5. and labels_value[1] <= 5.:
            #     label = 2 # 第三象限
            # elif labels_value[0] <= 5. and labels_value[1] > 5.:
            #     label = 2 # 第四象限
            # 将 60S 的数据按照时间窗口大小进行分割（data.shape=(32, 7680)）
            step = windows - overlapping # 每次移动的步长
            iterator_num = int((60 - windows) / step  + 1) # 划分时间片段总个数
            for iterator in range(iterator_num):
                data_slice = data[:,128*(iterator*step):128*(iterator*step+windows)]
                datas.append(data_slice)
                labels.append(label)
    print("Get sample data success!")
    print("Total sample number is: ", len(labels))
    print("label 1: {}  label 2: {}  label 3: {}  label 4: {}.".format(np.sum(np.array(labels)==1), 
                                                                       np.sum(np.array(labels)==2), 
                                                                       np.sum(np.array(labels)==3), 
                                                                       np.sum(np.array(labels)==4)))
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

def read_data(path=SAMPLES_PATH, people_list = list(range(0,32)), trial_list=list(range(0,40)), windows=4, overlapping=3, raw_data=False, sample_flag=None):
    # datas 和 labels 都是 list. datas 中的每一项是 shape=(32, 128*windows) 的数组
    datas, labels = get_samples_data(path, people_list, trial_list, windows, overlapping)
    datas_result = []
    for data in datas:
        data_list = []
        if not raw_data:
            # 数据预处理，提取特征(每一个样本处理后的结果是 numpy.narray 且 shape=(features, seq_length))
            for window in range(windows): # 1S 为一个单位提取特征
                features_list = []
                for i in range(32): # 依次处理 32 通道的 EEG 信号
                    X = data[i, 128*(window):128*((window+1))]
                    # theta, alpha, beta, gamma = data_filter(X, params) # 获取各个频率段的数据
                    features_list.append(differential_entropy(X, 4, 7))
                    features_list.append(differential_entropy(X, 8, 13))
                    features_list.append(differential_entropy(X, 14, 30))
                    features_list.append(differential_entropy(X, 31, 50))
                # _max = max(features_list)
                # _min = min(features_list)
                _max = 1.0
                _min = 0
                data_list.append((np.array(features_list).reshape(-1, 1) - _min)/(_max - _min)) # 0-1化处理
            datas_result.append(np.c_[tuple(data_list)]) # shape=(features, seq_length)
    if(raw_data):
        datas_result = datas
    del datas # 释放内存
    assert len(datas_result) == len(labels)
    if not raw_data and (sample_flag==True):
        np.save("./data_set/train_datas_features", datas_result)
        np.save("./data_set/train_label_features", labels)
    if not raw_data and (sample_flag==False):
        np.save("./data_set/test_datas_features", datas_result)
        np.save("./data_set/test_label_features", labels)
    return (datas_result, labels)
