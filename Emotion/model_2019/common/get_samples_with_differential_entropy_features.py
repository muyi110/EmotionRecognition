# -*- coding:UTF-8 -*-
#################################################################################
# 此模块获取提取差分熵特征后的样本数据. ------2018-12-06
#################################################################################
import os
import math
import numpy as np
from features_extraction import differential_entropy, filter_with_stft
from label_threshold_cluster import kmeans_cluster, get_samples_labels

SAMPLES_PATH = "../../samples/"

def _get_samples_data(people_list, trial_list, path, classify_object_name=0):
    '''
      classify_object_name = 0 ----> valence
      classify_object_name = 1 ----> arousal
    '''
    labels_all_people = get_samples_labels(SAMPLES_PATH) # 读取 32 个人的样本标签数据
    valence_list, arousal_list, _, _ = kmeans_cluster(labels_all_people) # 获取每一个人的标签阈值
    labels_dic = {}
    labels_dic['0'] = valence_list
    labels_dic['1'] = arousal_list
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
            if label_value[classify_object_name] >= labels_dic[str(classify_object_name)][people]:
                label = 0
            elif label_value[classify_object_name] < labels_dic[str(classify_object_name)][people]:
                label = 1
            filtered_array = np.zeros((128, 128*60))
            for i in range(32):  # 依次处理 32 通道数据
                # 获取 4 个子频段 EEG 数据
                theta, alpha, beta, gamma = filter_with_stft(eeg[i])
                filtered_array[4*i] = theta
                filtered_array[4*i+1] = alpha
                filtered_array[4*i+2] = beta
                filtered_array[4*i+3] = gamma
            datas.append(filtered_array)
            labels.append(label)
    print("Get data success!")
    print("Total samples number is: ", len(labels))
    print("label 0: {}, label 1: {}".format(np.sum(np.array(labels)==0), np.sum(np.array(labels)==1)))
    return datas, labels

def read_data(people_list, trial_list, path=SAMPLES_PATH, classify_object_name=0):
    # datas 和 labels 都是 list. datas 中的每一项都是 shape=(128, 128*60) 的 np.array
    datas, labels = _get_samples_data(people_list, trial_list, path, classify_object_name)
    datas_result = []
    for data in datas:
        data_list = []
        for window in range(60): # 有60s 的数据
            features_list = []
            for i in range(128):
                X = data[i, 128*window:128*(window+1)]
                if (i % 4) == 0:
                    features_list.append(differential_entropy(X, 4, 7))
                elif (i % 4) == 1:
                    features_list.append(differential_entropy(X, 8, 13))
                elif (i % 4) == 2:
                    features_list.append(differential_entropy(X, 14, 30))
                elif (i % 4) == 3:
                    features_list.append(differential_entropy(X, 31, 50))
            data_list.append(np.array(features_list).reshape(-1, 1))
        datas_result.append(np.c_[tuple(data_list)]) # 每一个元素 shape=(features, seq_length)=(128, 60)
    del datas
    assert len(datas_result) == len(labels)
    #_save_samples(datas_result, labels, people_list, classify_object_name)
    return (datas_result, labels)

def _save_samples(datas_result, labels, people_list, classify_object_name):
    if classify_object_name == 0:
        class_name = "valence_without_features_smooth"
    elif classify_object_name == 1:
        class_name = "arousal_without_features_smooth"
    # 针对单独一个人情况
    if len(people_list) == 1:
        print("save samples start: ")
        if not os.path.isdir(os.path.join("../common/samples_features/"+class_name, "s"+str(people_list[0]))):
            os.makedirs(os.path.join("../common/samples_features/"+class_name, "s"+str(people_list[0])))
        np.save("./samples_features/"+class_name+"/s"+str(people_list[0])+"/datas", datas_result)
        np.save("./samples_features/"+class_name+"/s"+str(people_list[0])+"/labels", labels)

if __name__ == "__main__":
    trial_list = list(range(0, 40))
    for number in range(32):
        people_list = [number]
        datas, labels = read_data(people_list, trial_list, path=SAMPLES_PATH, classify_object_name=0)
        assert(len(datas) == 40)
        assert(len(labels) == 40)
        assert(datas[0].shape == (128, 60))
    for number in range(32):
        people_list = [number]
        datas, labels = read_data(people_list, trial_list, path=SAMPLES_PATH, classify_object_name=1)
        assert(len(datas) == 40)
        assert(len(labels) == 40)
        assert(datas[0].shape == (128, 60))
