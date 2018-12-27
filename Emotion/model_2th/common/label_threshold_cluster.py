# -*- coding:UTF-8 -*-
############################################################################
# 针对每一个人的 40 个实验标签进行聚类（valence, arousal），得到每一个人的
# label 划分阈值（二分类）
############################################################################
import os
import numpy as np
from sklearn.cluster import KMeans

SAMPLE_PATH = "../../data_analysis/samples/"

def get_samples_labels(path):
    samples_dirs = os.listdir(path) #目录的顺序是随机的
    samples_dirs = sorted(samples_dirs)
    file_path = [os.path.join(path, samples_dirs[i]) for i in range(len(samples_dirs))]
    # labels 中的每个元素是二维数组
    labels = [np.loadtxt(file_path[i]+'/label.csv', delimiter=',', skiprows=0)[:,:2] 
                         for i in range(len(file_path))]
    return labels

def kmeans_cluster(input_data, k=2):
    labels = input_data
    assert (len(labels) == 32)
    kmeans = KMeans(n_clusters=k, random_state=None)
    label_center_list_v = []
    label_center_list_a = []
    numbers_temp_v = []
    numbers_temp_a = []
    for label in labels:
        assert(label.shape == (40, 2))
        cluster_result = kmeans.fit(label)
        center_v = cluster_result.cluster_centers_.mean(axis=0)[0]
        center_a = cluster_result.cluster_centers_.mean(axis=0)[1]
        numbers_temp_v.append((label[:,0:1]>=center_v).sum(axis=0)[0])
        numbers_temp_a.append((label[:,1:2]>=center_a).sum(axis=0)[0])
        label_center_list_v.append(float('%.4f' % center_v))
        label_center_list_a.append(float('%.4f' % center_a))
    return label_center_list_v, label_center_list_a, numbers_temp_v, numbers_temp_a

if __name__ == "__main__":
    labels = get_samples_labels(SAMPLE_PATH)
    valence_centers, arousal_centers, number_v, number_a = kmeans_cluster(labels)
    print("valence:")
    print(valence_centers)
    print(number_v)
    print("arousal:")
    print(arousal_centers)
    print(number_a)
