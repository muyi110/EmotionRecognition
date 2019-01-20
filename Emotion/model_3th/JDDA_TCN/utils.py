# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def test_for_train_samples(X_test, y_test):
    assert(y_test[:52].sum() == 0 or y_test[:52].sum() == 1)
    a = np.random.permutation(52)[:13]
    samples_select_list = []
    for i in range(8):
        samples_select_list += list(a + 52*i)
    samples_select_array = np.array(samples_select_list)
    assert(samples_select_array.shape == (104,))
    samples_select_X = X_test[samples_select_array]
    samples_select_y = y_test[samples_select_array]
    return samples_select_X, samples_select_y

def batch_generator(X, y, X_test, y_test, batch_size, seed):
    assert(X.shape[0] == X_test.shape[0]*4)
    X_test = np.vstack([X_test]*4)
    y_test = np.hstack([y_test]*4)
    # assert(X.shape[0] == X_test.shape[0]*4)
    # X_test = np.vstack([X_test, X_test, X_test, X_test])
    # y_test = np.hstack([y_test, y_test, y_test, y_test])
    assert(len(y) == len(y_test))
    np.random.seed(seed)
    num = X.shape[0]
    permutation = list(np.random.permutation(num))
    shuffled_X = X[permutation]
    shuffled_y = y[permutation]
    shuffled_X_test = X_test[permutation]
    shuffled_y_test = y_test[permutation]
    
    num_complete_minibatches = num // (batch_size)
    minibatchs = []
    for k in range(num_complete_minibatches):
        X_s = shuffled_X[k*(batch_size):(k+1)*(batch_size)]
        y_s = shuffled_y[k*(batch_size):(k+1)*(batch_size)]
        X_t = shuffled_X_test[k*(batch_size):(k+1)*(batch_size)]
        y_t = shuffled_y_test[k*(batch_size):(k+1)*(batch_size)]
        # X_batch = np.vstack([X_s, X_t])
        # y_batch = np.hstack([y_s, y_t])
        mnibatch = (X_s, y_s, X_t, y_t)
        # assert(len(y_batch) == batch_size)
        assert(X_s.shape == (batch_size, 9, 128))
        assert(X_t.shape == (batch_size, 9, 128))
        minibatchs.append(mnibatch)
    return minibatchs

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
