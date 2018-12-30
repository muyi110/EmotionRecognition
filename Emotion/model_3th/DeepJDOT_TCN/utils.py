# -*- coding:UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def _shuffle_aligned_list(datas):
    '''
    data: 一个列表，存放训练数据和对应的标签
    '''
    num = datas[0].shape[0]  # 获取样本的个数
    p = np.random.permutation(num)
    return [d[p] for d in datas]

def batch_generator(data, batch_size, shuffle=True):
    if shuffle:
        data = _shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > len(data[0]):
            batch_count = 0
            if shuffle:
                data = _shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]

def plot_embedding(X, y, d, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X-x_min) / (x_max - x_min)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]), color=plt.cm.bwr(d[i] / 1.), fontdict={"weight":"bold", "size":9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()
