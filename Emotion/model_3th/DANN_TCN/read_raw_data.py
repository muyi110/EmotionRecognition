# -*- coding:UTF-8 -*-
import sys
sys.path.append("../common/")
import os
import numpy as np
from label_threshold_cluster import kmeans_cluster, get_samples_labels

SAMPLES_PATH = "../../samples/"
def read_data(people_list, trial_list, path, classify_object_name=0):
    labels_all_people = get_samples_labels(SAMPLES_PATH)
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
            # 对 EEG 进行 0-1 化处理，验证通道轴
            _min = eeg.min(axis=0)
            _max = eeg.max(axis=0)
            data = (eeg - _min) / (_max - _min)
            step = 1 # 每次移动的步长
            iterator_num = int((60 - 9) / step  + 1) # 划分时间片段总个数
            for iterator in range(iterator_num):
                data_slice = data[:,128*(iterator*step):128*(iterator*step+9)]
                datas.append(data_slice)
                labels.append(label)
    print("datas shape: ", np.array(datas).shape)
    return (datas, labels)
