# -*- coding:UTF-8 -*-
###################################################################
# 读取外周生理信号
###################################################################
import os
import math
import numpy as np
from label_threshold_cluster import kmeans_cluster, get_samples_labels

SAMPLES_PATH = "../../samples/"

def get_data(people_list, trial_list, path=SAMPLES_PATH, classify_object_name=0):
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
            perip_data = np.loadtxt(file_path[people]+"/trial_"+str(trial+1)+".csv", delimiter=',', 
                             skiprows=0, dtype=np.float32)
            perip_data = perip_data[32:, 128*3:] # 取后 60s 的外周生理信号
            label_value = np.loadtxt(file_path[people]+"/label.csv", delimiter=",", skiprows=0,
                                     dtype=np.float32)[trial, :2]
            # label_value[0]-->valence, label_value[1]-->arousal
            if label_value[classify_object_name] >= labels_dic[str(classify_object_name)][people]:
                label = 0
            elif label_value[classify_object_name] < labels_dic[str(classify_object_name)][people]:
                label = 1
            datas.append(perip_data)
            labels.append(label)
    print("Get data success!")
    print("Total samples number is: ", len(labels))
    print("label 0: {}, label 1: {}".format(np.sum(np.array(labels)==0), np.sum(np.array(labels)==1)))
    return datas, labels

if __name__ == "__main__":
    people_list = [0]
    trail_list = list(range(40))
    datas, labels = get_data(people_list, trail_list, SAMPLES_PATH, 0)
    print("datas shape: {}".format(np.array(datas).shape))
    print("labels shape: {}".format(np.array(labels).shape))
