# -*- coding:UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def batch_generator(X, y, batch_size, seed):
    assert(X.shape[0] == 52*32)
    assert(y.shape == (52*32,))
    assert(np.sum(y[:52]) == 0 or np.sum(y[:52]) == 52)
    X_class_one = X[y==0]
    X_class_two = X[y==1]
    y_class_one = y[y==0]
    y_class_two = y[y==1]
    assert(X_class_one.shape[0] + X_class_two.shape[0] == 32*52)
    assert(X_class_one.shape == (np.sum(y==0), 9, 128)) # 128->60
    np.random.seed(seed)
    permutation = list(np.random.permutation(52))
    
    num_complete_minibatches = 52 // (batch_size//32)
    minibatchs = []
    for k in range(num_complete_minibatches):
        X_s = permutation[k*(batch_size//32):(k+1)*(batch_size//32)]
        y_s = permutation[k*(batch_size//32):(k+1)*(batch_size//32)]
        temp_list_X_one = []
        temp_list_y_one = []
        for i in range(X_class_one.shape[0]//52):
            temp_list_X_one.append(X_class_one[list(np.array(X_s)+52*i)])
            temp_list_y_one.append(y_class_one[list(np.array(y_s)+52*i)])
        class_one_X = np.vstack(tuple(temp_list_X_one))
        class_one_y = np.hstack(tuple(temp_list_y_one))
        temp_list_X_two = []
        temp_list_y_two = []
        for i in range(X_class_two.shape[0]//52):
            temp_list_X_two.append(X_class_two[list(np.array(X_s)+52*i)])
            temp_list_y_two.append(y_class_two[list(np.array(y_s)+52*i)])
        class_two_X = np.vstack(tuple(temp_list_X_two))
        class_two_y = np.hstack(tuple(temp_list_y_two))
        X_batch = np.vstack([class_one_X, class_two_X])
        y_batch = np.hstack([class_one_y, class_two_y])
        mnibatch = (X_batch, y_batch)
        assert(X_batch.shape == (batch_size, 9, 128)) # 128->60
        assert(y_batch.shape == (batch_size,))
        minibatchs.append(mnibatch)
    return minibatchs, X_class_one.shape[0]//52
def plot_embedding(X, y, d, people, number, dataPath, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X-x_min) / (x_max - x_min)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]), color=plt.cm.bwr(d[i] / 1.), fontdict={"weight":"bold", "size":9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig(dataPath+str(people)+str(number)+".png")
    #plt.show()
