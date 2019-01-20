# -*- coding:UTF-8 -*-
import numpy as np
from scipy.io import loadmat

def load_mnist():
    mnist_data = loadmat("../data/mnist_data.mat")
    mnist_train = mnist_data['train_28'].astype(np.float32)
    mnist_test = mnist_data['test_28'].astype(np.float32)
    mnist_labels_train = mnist_data['label_train']
    mnist_labels_test = mnist_data['label_test']

    train_label = np.argmax(mnist_labels_train, axis=1)
    inds = np.random.permutation(mnist_train.shape[0])
    mnist_train = mnist_train[inds]
    train_label = train_label[inds]
    test_label = np.argmax(mnist_labels_test, axis=1)
    
    return mnist_train, train_label, mnist_test, test_label
