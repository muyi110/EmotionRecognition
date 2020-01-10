#! /usr/bin/env python3
# -*- coding:UTF-8 -*-
import sys
sys.path.append("../common")
import numpy as np
from read_data import read_data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

if __name__ == "__main__":
    people_num_list = list(range(0, 32))
    for people_num_ in people_num_list:
        datas, labels = read_data(people_list=[people_num_], windows=9, overlapping=8, classify_object_name=0, 
                                  mv_flag=True, lds_flag=False)
        datas = np.array(datas).mean(axis=2)
        labels = np.array(labels)
        print("datas shape: ", datas.shape)
        # 特征可视化
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
        tsne.fit_transform(datas)
        X = tsne.embedding_
        plt.subplot(8, 4, people_num_ + 1)
        plt.scatter(X[:, 0], X[:, 1], s=10, c=labels)
        plt.xticks([])
        plt.yticks([])
        plt.title(people_num_+1)
        plt.subplots_adjust(wspace=0.1, left=0.03, right=0.99, top=0.95, bottom=0.01)
    plt.savefig("./1.png")
    #plt.show()
    plt.clf()
    for people_num_ in people_num_list:
        datas, labels = read_data(people_list=[people_num_], windows=9, overlapping=8, classify_object_name=0, 
                                  mv_flag=True, lds_flag=False)
        datas = np.array(datas).mean(axis=2)
        labels = np.array(labels)
        labels_new = [[i]*52 for i in range(40)]
        labels_new = np.array(labels_new).reshape(-1)
        print("datas shape: ", datas.shape)
        # 特征可视化
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
        tsne.fit_transform(datas)
        X = tsne.embedding_
        plt.subplot(8, 4, people_num_ + 1)
        plt.scatter(X[:, 0], X[:, 1], s=10, c=labels_new)
        plt.xticks([])
        plt.yticks([])
        plt.title(people_num_+1)
        plt.subplots_adjust(wspace=0.1, left=0.03, right=0.99, top=0.95, bottom=0.01)
    plt.savefig("./2.png")
    #plt.show()
