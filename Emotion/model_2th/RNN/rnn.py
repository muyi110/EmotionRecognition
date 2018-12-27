#!/usr/bin/env python3
# -*- coding=UTF-8 -*-
import time
import sys
sys.path.append("../common/")
import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from read_data import read_data, index_generator
from sklearn.svm import SVC

# 构建 RNN 模型类，为了兼容 scikit-learning 的 RandomizedSearchCV 类，后续可能实现超参数搜索
class RNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 n_steps = 9,
                 n_inputs = 128, # 可以理解为输入的特征数
                 n_neurons = 500,# 每个 cell 包含的神经元个数
                 n_outputs = 2, # 4 分类
                 learning_rate = 0.001,
                 random_state = None, 
                 batch_size = 64,
                 dropout = 0.0,
                 optimizer_class = tf.train.AdamOptimizer):
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.batch_size = batch_size
        self.dropout = dropout
        self.optimizer_class = optimizer_class
        self._session = None
    def _rnn(self, inputs, multi_layer=False):
        lstm_cells_1 = tf.nn.rnn_cell.LSTMCell(num_units=self.n_neurons) # 构建 LSTM cell
        if multi_layer:
            #lstm_cells_1 = tf.contrib.rnn.DropoutWrapper(lstm_cells_1, input_keep_prob=1-self.dropout)
            lstm_cells_2 = tf.contrib.rnn.BasicLSTMCell(num_units=50) # 构建第二层 LSTM cell
            #lstm_cells_2 = tf.contrib.rnn.DropoutWrapper(lstm_cells_2, input_keep_prob=1-self.dropout)
            lstm_cells_3 = tf.contrib.rnn.BasicLSTMCell(num_units=50) # 构建第三层 LSTM cell
            #lstm_cells_3 = tf.contrib.rnn.DropoutWrapper(lstm_cells_3, input_keep_prob=1-self.dropout)
            multi_cell_cell = tf.contrib.rnn.MultiRNNCell([lstm_cells_1, lstm_cells_2, lstm_cells_3])
        else:
            multi_cell_cell = lstm_cells_1
        outputs, states = tf.nn.dynamic_rnn(multi_cell_cell, inputs, dtype=tf.float32)
        return outputs, states
    def _bulid_graph(self, multi_cell_cell):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)
        X = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        y = tf.placeholder(tf.int32, [None])
        is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
        rnn_outputs = self._rnn(X, multi_cell_cell) # 获取 RNN 网络最后的输出状态
        if multi_cell_cell:
            top_layer_h_state = rnn_outputs[1][-1][1]  # 获取顶层的 RNN 输出状态(短期状态)
        else:
            top_layer_h_state = tf.layers.flatten(rnn_outputs[0])
            # top_layer_h_state = rnn_outputs[1][1]
            print("states: ", top_layer_h_state)
            print("outputs: ", rnn_outputs[0])
        # 对 RNN 的输出应用 dropout
        # top_layer_h_state = tf.contrib.layers.dropout(top_layer_h_state, 1-self.dropout, is_training=is_training)
        out = top_layer_h_state
        logits = tf.layers.dense(out, self.n_outputs, 
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), 
                                 name="logits") # 全连接层默认是没有激活函数的
        Y_proba = tf.nn.softmax(logits, name="Y_proba")
        # 计算交叉熵
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
        # 构建优化器节点
        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)
        # 构建计算准确率节点
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        # 构建全局初始化节点和模型保存节点
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self._X, self._y = X, y
        self._Y_proba, self._loss = Y_proba, loss
        self._training_op, self._accuracy = training_op, accuracy
        self._init, self._saver = init, saver
        self._is_training = is_training
    def close_session(self):
        if self._session:
            self._session.close()
    def fit(self, X, y, n_epochs=100, X_test=None, y_test=None, multi_cell_cell=False):
        self.close_session()
        
        self.classes_ = np.unique(y)
        self.class_to_index_ = {label: index for index, label in enumerate(self.classes_)}
        y = np.array([self.class_to_index_[label] for label in y], dtype=np.int32)
        self.y_test_classes_ = np.unique(y_test)
        self.y_test_class_to_index_ = {label: index for index, label in enumerate(self.y_test_classes_)}
        y_test = np.array([self.y_test_class_to_index_[label] for label in y_test], dtype=np.int32)

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._bulid_graph(multi_cell_cell) # 构建模型
        # 开始训练阶段
        seed = 0
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            sess.run(self._init)
            for epoch in range(n_epochs):
                seed += 1
                start_time = time.time()
                for X_batch_index, y_batch_index in index_generator(len(y), self.batch_size, seed=seed):
                    X_batch = X[X_batch_index]
                    y_batch = y[y_batch_index]
                    sess.run(self._training_op, feed_dict={self._X: X_batch, self._y: y_batch, self._is_training:True})
                total_loss = 0 
                total_acc = 0
                if epoch % 1 == 0:
                    for i in range(len(y) // 8):
                        X_batch = X[i*8:(i+1)*8,:,:]
                        y_batch = y[i*8:(i+1)*8]
                        loss_train, acc_train = sess.run([self._loss, self._accuracy], 
                                                        feed_dict={self._X:X_batch, self._y:y_batch, self._is_training:False})
                        total_loss += loss_train
                        total_acc += acc_train
                        end_time = time.time()
                    print("{}\ttraining loss: {:.6f}\t|  training accuracy: {:.2f}% | time: {:.2f}s".format(epoch, 
                          total_loss/(len(y)//8), (total_acc / (len(y)//8))*100, end_time-start_time))
                if X_test is not None and y_test is not None and epoch % 1 == 0:
                    total_acc_test = 0
                    total_loss_test = 0
                    for i in range(len(y_test) // 8):
                        X_batch_test = X_test[i*8:(i+1)*8, :, :]
                        y_batch_test = y_test[i*8:(i+1)*8]
                        loss_test, acc_test = sess.run([self._loss, self._accuracy], 
                                feed_dict={self._X:X_batch_test, self._y:y_batch_test, self._is_training:False})
                        total_acc_test += acc_test
                        total_loss_test += loss_test
                    print("Test accuracy: {:.4f}%\t  Test loss: {:.6f}".format((total_acc_test / (len(y_test) // 8))*100, total_loss_test/(len(y_test) // 8)))
                    #loss_test, acc_test = sess.run([self._loss, self._accuracy], 
                    #                                feed_dict={self._X:X_test, self._y:y_test, self._is_training:False})
                    #print("Test accuracy: {:.4f}%\t Test loss: {:.6f}".format(acc_test*100, loss_test))
            return self
    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._Y_proba.eval(feed_dict={self._X: X, self._is_training:False})
    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([self.classes_[class_index] for class_index in class_indices], np.int32).reshape(-1)
    def save(self, path):
        self._saver.save(self._session, path)

if __name__ == "__main__":
    # 开始将数据集划分为训练集和测试集
    np.random.seed(42)
    permutation = list(np.random.permutation(40)) # 将数据随机打乱
    train_index = [7]
    test_index = [7]
    train_trial_list = list(permutation[:32])
    test_trial_list = list(permutation[32:])
    # 获取生理信号数据
    # datas_train, train_labels = read_data(people_list=train_index, classify_object_name=0, train_flag=True,
    #                                       trial_list=train_trial_list, windows=9, overlapping=8)
    # datas_test, test_labels = read_data(people_list=test_index, classify_object_name=0, train_flag=False, 
    #                                     trial_list=test_trial_list, windows=9, overlapping=8)
    datas_train = np.load("../common/samples_single_people/s31/train_datas.npy") 
    train_labels = np.load("../common/samples_single_people/s31/train_labels.npy") 
    datas_test = np.load("../common/samples_single_people/s31/test_datas.npy")  
    test_labels = np.load("../common/samples_single_people/s31/test_labels.npy")  
    
    datas_train = np.array(datas_train)
    train_labels = np.array(train_labels)
    datas_test = np.array(datas_test)
    test_labels = np.array(test_labels)
    print("train data set number: ", len(train_labels))
    print("train datas shape: ", datas_train.shape)
    print("test data set number: ", len(test_labels))
    print("test datas shape: ", datas_test)
    print("train label 1: ", sum(train_labels==1), "train label 2: ", sum(train_labels==2))
    print("test label 1: ", sum(test_labels==1), "test label 2: ", sum(test_labels==2))

    datas_train = datas_train.transpose((0,2,1))
    datas_test = datas_test.transpose((0,2,1))
    print("train number: ", len(train_labels))
    print(datas_train.shape, train_labels.shape)
    print("test number: ", len(test_labels))
    print(datas_test.shape, test_labels.shape)
    # # 尝试 SVM 
    # datas_train = datas_train.mean(axis=1)
    # datas_test = datas_test.mean(axis=1)
    # svm_clf = SVC(decision_function_shape="ovr", gamma=0.0001, C=0.01, kernel="rbf", random_state=42)
    # svm_clf.fit(datas_train, train_labels-1)
    # print("accuracy: ", accuracy_score(train_labels-1,svm_clf.predict(datas_train)))
    # print("accuracy: ", accuracy_score(test_labels-1,svm_clf.predict(datas_test)))
    rnn = RNNClassifier(random_state=42, learning_rate=1e-5, dropout=0.0)
    rnn.fit(X=datas_train, y=train_labels, n_epochs=351, X_test=datas_test, y_test=test_labels, multi_cell_cell=False)
