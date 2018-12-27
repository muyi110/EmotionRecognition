# -*- coding:UTF-8 -*-
#################################################################################
# 此模块获取样本数据. ------2018-11-26
#################################################################################
import os
import math
import numpy as np
from features_extraction import differential_entropy, filter_with_stft

SAMPLES_PATH = "../../samples/"

def index_generator(num_examples, batch_size, seed=0):
    '''
      此函数用于生成 batch 的索引
    '''
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

def _get_samples_data(people_list, trial_list, path, windows=9, overlapping=8, classify_object_name=0):
    '''
      classify_object_name = 0 ----> valence
      classify_object_name = 1 ----> arousal
    '''
    samples_dirs = os.listdir(path) # 目录的顺序是随机的
    samples_dirs = sorted(samples_dirs)
    file_path = [os.path.join(path, samples_dirs[i]) for i in range(len(samples_dirs))]
    datas = [] # 获取最终的样本
    labels = [] # 对应的样本
    for people in people_list:
        for trial in trial_list:
            eeg = np.loadtxt(file_path[people]+"/trial_"+str(trial+1)+".csv", delimiter=',', 
                             skiprows=0, dtype=np.float32)
            eeg = eeg[:32, 128*3:] # 取后 60s 的 EEG 信号
            label_value = np.loadtxt(file_path[people]+"/label.csv", delimiter=",", skiprows=0,
                                     dtype=np.float32)[trial, :2]
            # label_value[0]-->valence, label_value[1]-->arousal
            if label_value[classify_object_name] >= 5.0:
                label = 0
            elif label_value[classify_object_name] < 5.0:
                label = 1
            step = windows - overlapping # 每次移动的步长
            iterator_num = int((60 - windows) / step + 1) # 划分时间片段总个数
            for iterator in range(iterator_num):
                data_slice = eeg[:, 128*(iterator*step):128*(iterator*step+windows)]
                datas.append(data_slice)
                labels.append(label)
    print("Get data success!")
    print("Total samples number is: ", len(labels))
    print("label 0: {}, label 1: {}".format(np.sum(np.array(labels)==0), np.sum(np.array(labels)==1)))
    return datas, labels

def read_data(people_list, trial_list, path=SAMPLES_PATH, windows=9, overlapping=8, 
              classify_object_name=0, train_flag=True):
    # datas 和 labels 都是 list. datas 中的每一项都是 shape=(32, 128*windows) 的 np.array
    datas, labels = _get_samples_data(people_list, trial_list, path, windows, overlapping, classify_object_name)
    datas_result = []
    for data in datas:
        data_list = []
        for window in range(windows):
            features_list = []
            for i in range(32):
                X = data[i, 128*window:128*(window+1)]
                features_list.append(differential_entropy(X, 4, 7))
                features_list.append(differential_entropy(X, 8, 13))
                features_list.append(differential_entropy(X, 14, 30))
                features_list.append(differential_entropy(X, 31, 50))
            data_list.append(np.array(features_list).reshape(-1, 1))
        datas_result.append(np.c_[tuple(data_list)]) # 每一个元素 shape=(features, seq_length)=(128, 9)
    del datas
    assert len(datas_result) == len(labels)
    _save_samples(datas_result, labels, people_list, train_flag)
    return (datas_result, labels)

def _save_samples(datas_result, labels, people_list, train_flag):
    # 针对单独一个人情况
    if len(people_list) == 1:
        if not os.path.isdir(os.path.join("./samples_single_people", "s"+str(people_list[0]))):
            os.makedirs(os.path.join("./samples_single_people", "s"+str(people_list[0])))
        if train_flag:
            np.save("./samples_single_people/s"+str(people_list[0])+"/train_datas", datas_result)
            np.save("./samples_single_people/s"+str(people_list[0])+"/train_labels", labels)
        else:
            np.save("./samples_single_people/s"+str(people_list[0])+"/test_datas", datas_result)
            np.save("./samples_single_people/s"+str(people_list[0])+"/test_labels", labels)
    if len(people_list) > 1:
        if train_flag:
            np.save("./samples_all_people"+"/train_datas", datas_result)
            np.save("./samples_all_people"+"/train_labels", labels)
        else:
            np.save("./samples_all_people"+"/test_datas", datas_result)
            np.save("./samples_all_people"+"/test_labels", labels)
