# -*- coding:UTF-8 -*-
#################################################################################
# 此模块获取样本数据. ------2018-11-26
#################################################################################
import os
import math
import numpy as np
from features_smooth_mv import moving_average, moving_average_mid
from features_smooth_lds import linear_dynamical_systems

SAMPLES_PATH = "./samples_features/valence_without_features_smooth/"

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

def _get_samples_data(people_list, path, windows=9, overlapping=8):
    samples_dirs = os.listdir(path) # 目录的顺序是随机的
    samples_dirs = sorted(samples_dirs)
    file_path = [os.path.join(path, samples_dirs[i]) for i in range(len(samples_dirs))]
    datas_list = [] # 获取最终的样本
    labels_list = [] # 对应的样本
    for people in people_list:
        datas = np.load(file_path[people]+"/datas.npy")
        labels = np.load(file_path[people]+"/labels.npy")
        assert (len(datas) == 40)
        assert (len(labels) == 40)
        for trial_num in range(len(datas)):
            data = datas[trial_num]
            label = labels[trial_num]
            assert (data.shape == (128, 60))
            step = windows - overlapping # 每次移动的步长
            iterator_num = int((60 - windows) / step + 1) # 划分时间片段总个数
            for iterator in range(iterator_num):
                data_slice = data[:, iterator*step:iterator*step+windows]
                datas_list.append(data_slice)
                labels_list.append(label)
    print("Get data success!")
    print("Total samples number is: ", len(labels))
    print("label 0: {}, label 1: {}".format(np.sum(np.array(labels)==0), np.sum(np.array(labels)==1)))
    return datas_list, labels_list

def read_data(people_list, windows=9, overlapping=8, classify_object_name=0, mv_flag=False, lds_flag=True):
    # datas 和 labels 都是 list. datas 中的每一项都是 shape=(128, 128*windows) 的 np.array
    if classify_object_name == 0:
        path = "../common/samples_features/valence_without_features_smooth/"
    elif classify_object_name == 1:
        path = "../common/samples_features/arousal_without_features_smooth/"
    datas, labels = _get_samples_data(people_list, path, windows, overlapping)
    assert len(datas) == len(labels)
    if mv_flag == True:
        # 滑动平均滤波
        datas_result = moving_average(datas, windows, windows=20)
        # datas_result = moving_average_mid(datas, windows, windows=20)
    elif lds_flag == True:
        # LDS 滤波
        model = np.load("../common/lds_model_params/model_params_53.npy").item()
        datas_result = linear_dynamical_systems(datas, windows, model)
    else:
        # 不做滤波处理
        datas_result = datas
    return (datas_result, labels)
