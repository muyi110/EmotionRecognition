# -*- coding:UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def batch_generator_new(data, batch_size, shuffle=True, seed=0):
    '''
    一个 batch 由 batch_size/2 个source 样本和 batch_size/2 个target 样本组成
    source 样本和 target 样本的各个类别比例相同，目前设置为 1:1
    '''
    num_each_class = int(batch_size / 4)
    X = data[0]
    y = data[1]
    X_test = data[2]
    y_test = data[3]
    X_class_one = X[list(y == 0)]
    X_class_two = X[list(y == 1)]
    y_class_one = y[list(y == 0)]
    y_class_two = y[list(y == 1)]

    X_test_class_one = X_test[list(y_test == 0)]
    X_test_class_two = X_test[list(y_test == 1)]
    y_test_class_one = y_test[list(y_test == 0)]
    y_test_class_two = y_test[list(y_test == 1)]
    batch_count_X0 = 0
    batch_count_X1 = 0
    batch_count_X_test1 = 0
    batch_count_X_test0 = 0
    while True:
        np.random.seed(seed=seed)
        X_p0 = list(np.random.permutation(len(X_class_one)))
        X_p1 = list(np.random.permutation(len(X_class_two)))
        X_test_p0 = list(np.random.permutation(len(X_test_class_one)))
        X_test_p1 = list(np.random.permutation(len(X_test_class_two)))
        if batch_count_X0 * num_each_class + num_each_class > len(X_class_one):
            seed += 1
            batch_count_X0 = 0
            np.random.seed(seed=seed)
            X_p0 = list(np.random.permutation(len(X_class_one)))
        if batch_count_X1 * num_each_class + num_each_class > len(X_class_two):
            seed += 1
            batch_count_X1 = 0
            np.random.seed(seed=seed)
            X_p1 = list(np.random.permutation(len(X_class_two)))
        if batch_count_X_test0 * num_each_class + num_each_class > len(X_test_class_one):
            seed += 1
            batch_count_X_test0 = 0
            np.random.seed(seed=seed)
            X_test_p0 = list(np.random.permutation(len(X_test_class_one)))
        if batch_count_X_test1 * num_each_class + num_each_class > len(X_test_class_two):
            seed += 1
            batch_count_X_test1 = 0
            np.random.seed(seed=seed)
            X_test_p1 = list(np.random.permutation(len(X_test_class_two)))
            
        X_start0 = batch_count_X0 * num_each_class
        X_end0 = X_start0 + num_each_class
        X_start1 = batch_count_X1 * num_each_class
        X_end1 = X_start1 + num_each_class

        X_test_start0 = batch_count_X_test0 * num_each_class
        X_test_end0 = X_test_start0 + num_each_class
        X_test_start1 = batch_count_X_test1 * num_each_class
        X_test_end1 = X_test_start1 + num_each_class

        a = X_class_one[X_p0[X_start0:X_end0]]
        a_l = y_class_one[X_p0[X_start0:X_end0]]
        b = X_class_two[X_p1[X_start1:X_end1]]
        b_l = y_class_two[X_p1[X_start1:X_end1]]
        c = X_test_class_one[X_test_p0[X_test_start0:X_test_end0]]
        c_l = y_test_class_one[X_test_p0[X_test_start0:X_test_end0]]
        d = X_test_class_two[X_test_p1[X_test_start1:X_test_end1]]
        d_l = y_test_class_two[X_test_p1[X_test_start1:X_test_end1]]
        X_batch = np.vstack([a, b, c, d])
        y_batch = np.hstack([a_l, b_l, c_l, d_l])
        batch_count_X0 += 1
        batch_count_X1 += 1
        batch_count_X_test1 += 1
        batch_count_X_test0 += 1
        yield (X_batch, y_batch)

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
