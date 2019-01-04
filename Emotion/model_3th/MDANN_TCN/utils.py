# -*- coding:UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def batch_generator(data, batch_size):
    X = data[0]
    y = data[1]
    num_each_domain = batch_size // 32
    assert(y[:52].sum() == 0 or y[:52].sum() == 52)
    assert(y[-52:].sum() == 0 or y[-52:].sum() == 52)
    p = np.random.permutation(52)
    batch_count = 0
    while True:
        if batch_count * num_each_domain + num_each_domain > 52:
            batch_count = 0
            p = np.random.permutation(52)
        start = batch_count * num_each_domain
        end = start + num_each_domain
        batch_count += 1
        X_batch_list = []
        y_batch_list = []
        for i in range(32):
            X_batch_list.append(X[p[start:end]+52*i])
            y_batch_list.append(y[p[start:end]+52*i])
        X_batch = np.r_[tuple(X_batch_list)]
        y_batch = np.r_[tuple(y_batch_list)]
        assert(X_batch.shape == (batch_size, 9, 128))
        assert(y_batch.shape == (batch_size,))
        X_batch_0 = X_batch[y_batch == 0]
        X_batch_1 = X_batch[y_batch == 1]
        y_batch_0 = y_batch[y_batch == 0]
        y_batch_1 = y_batch[y_batch == 1]
        X_batch = np.r_[X_batch_0, X_batch_1]
        y_batch = np.r_[y_batch_0, y_batch_1]
        num_0 = (y_batch==0).sum()
        assert(y_batch[:num_0].sum() == 0 and y_batch[num_0:].sum() == batch_size-num_0)
        yield (X_batch, y_batch, num_0)

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
