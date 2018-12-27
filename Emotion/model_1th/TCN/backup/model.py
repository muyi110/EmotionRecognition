#!/usr/bin/env python3
# -*- coding:UTF-8 -*-
import numpy as np
import tensorflow as tf
import time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from tcn import TCN
from data import read_data, index_generator

# 构建 TCN 模型类，为了兼容 scikit-learning 的 RandomizedSearchCV 类，后续可能实现超参数搜索
class TCNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, num_channels, 
                 sequence_length, 
                 kernel_size, 
                 dropout, 
                 batch_size=16, 
                 in_channels=1, 
                 random_state=None, 
                 learning_rate=0.001, 
                 optimizer_class=tf.train.AdamOptimizer):
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.batch_size = batch_size
        self.random_state = random_state
        self.in_channels = in_channels
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self._session = None
    def _TCN(self, inputs, n_outputs):
        '''构建 TCN 模型'''
        outputs = TCN(inputs, n_outputs, self.num_channels, 
                      self.sequence_length, self.kernel_size, self.dropout)
        return outputs
    def _bulid_graph(self, n_outputs):
        '''构建计算图'''
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)
        inputs = tf.placeholder(tf.float32, 
                                shape=(None, self.sequence_length, self.in_channels), name="inputs")
        labels = tf.placeholder(tf.int32, shape=(None), name="labels")
        #self._training = tf.placeholder_with_default(True, shape=(), name="training")
        tcn_outputs = self._TCN(inputs, n_outputs)
        predictions = tf.nn.softmax(tcn_outputs, name="predictions")
        # 计算交叉熵
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=tcn_outputs)
        loss = tf.reduce_mean(xentropy, name="loss")
        # 构建优化器节点
        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)
        # 构建计算准确率节点
        correct = tf.nn.in_top_k(tcn_outputs, labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        # 构建全局初始化节点和模型保存节点
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self._X, self._y = inputs, labels
        self._predictions, self._loss = predictions, loss
        self._training_op, self._accuracy = training_op, accuracy
        self._init, self._saver = init, saver
    def close_session(self):
        if self._session:
            self._session.close()
    def _get_model_params(self):
        '''获取所有变量值，用于 early stopping ,faster than saving to disk'''
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)# 获取一个 list 包含所有的变量
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}
    def _restore_model_params(self, model_params):
        gvar_names = list(model_params.keys())
        # 获取被给名字的操作(op)
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign") for gvar_name in gvar_names}
        # inputs 是tf.Operation 的属性. The list of Tensor objects representing the data inputs of this op
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        # 由于 key 是 tensor ，所以 value 会替换为 key 对应的 tensor. 具体参考官网 tf.Session.run
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)
    def fit(self, X, y, n_epochs, X_valid=None, y_valid=None):
        '''Fit the model to the training set. If X_valid and y_valid are provided, use early stopping'''
        self.close_session()
        self.classes_ = np.unique(y)
        n_outputs = len(self.classes_) # 获取输出的类别数
        self.class_to_index_ = {label:index for index, label in enumerate(self.classes_)}
        y = np.array([self.class_to_index_[label] for label in y], dtype=np.int32)
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._bulid_graph(n_outputs) # 构建计算模型
        # 下面几个变量用于 early stopping
        max_check_without_progress = 20
        checks_without_progress = 0
        best_loss = np.infty
        best_params = None
        # 开始训练阶段
        seed = 0
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            sess.run(self._init)
            for epoch in range(n_epochs):
                seed += 1
                start_time = time.time()
                #for iteration in range(int(np.ceil(len(y)/self.batch_size))):
                for X_batch_index, y_batch_index in index_generator(len(y), self.batch_size, seed=seed):
                    X_batch = X[X_batch_index]
                    y_batch = y[y_batch_index]
                    sess.run(self._training_op, feed_dict={self._X: X_batch, self._y: y_batch})
                    #print("{}\ttraining batch loss: {:.6f}".format(seed, loss_val))

                # 下面用于 early stopping
                if X_valid is not None and y_valid is not None:
                    loss_val, acc_val = sess.run([self._loss, self._accuracy], 
                                                 feed_dict={self._X:X_valid, self._y:y_valid})
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_params = self._get_model_params()
                        checks_without_progress = 0
                    else:
                        checks_without_progress += 1
                    print("{}\tValidation loss: {.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(epoch, 
                          loss_val, best_loss, acc_val*100))
                    if checks_without_progress >= max_check_without_progress:
                        print("Early stopping!")
                else:
                    total_loss = 0 
                    total_acc = 0
                    for i in range(len(y) // 8):
                        X_batch = X[i*8:(i+1)*8,:,:]
                        y_batch = y[i*8:(i+1)*8]
                        loss_train, acc_train = sess.run([self._loss, self._accuracy], 
                                                      feed_dict={self._X:X_batch, self._y:y_batch})
                        total_loss += loss_train
                        total_acc += acc_train
                        end_time = time.time()
                    print("{}\ttraining loss: {:.6f}\t|  training accuracy: {:.2f}% | time: {:.2f}s".format(epoch, 
                          total_loss/(len(y)//4), (total_acc / (len(y)//4))*100, end_time-start_time))
            if best_params:
                self._restore_model_params(best_params)
            return self
    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._predictions.eval(feed_dict={self._X: X})
    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([[self.classes_[class_index]] for class_index in class_indices], np.int32)
    def save(self, path):
        self._saver.save(self._session, path)

if __name__ == "__main__":
    n_classes = 4 # 4 分类问题
    input_channels = 32 # 输入通道数 32
    seq_length = int(60*128*32 / input_channels) # 序列的长度
    dropout = 0.2
    learning_rate=0.0001
    num_channels = [50, 50, 50, 50, 50, 
                    10, 10, 10, 10, 10] # 有多少层，及每一层包含的神经元个数（这里的一层指一个 block）
    kernel_size = 9   # 卷积核大小  
    batch_size = 4

    # 获取生理信号数据(训练集)
    datas, labels = read_data(train=True, test=False, input_datas_norm=True, seed=42)
    # 数据处理，目前只用到 32 通道的 EEG 信号
    datas_eeg = np.array(datas)[:,0:32,:] # datas_result 的 shape=(880, 32, 7680)
    # datas_eeg = tf.transpose(datas_eeg, perm=[0, 2, 1]) # 将 datas_result 的形状变为 shape=(880, 7680, 32)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     datas_eeg = datas_eeg.eval() # 将 tensor 转为 numpy.array
    datas_eeg = datas_eeg.transpose((0,2,1)) # 将 datas_result 的形状变为 shape=(880, 7680, 32)
    labels = np.array(labels).reshape(-1)
    # 开始构建TCN 模型实例
    tcn = TCNClassifier(num_channels=num_channels, sequence_length = seq_length, kernel_size=kernel_size, 
                        dropout=dropout, batch_size=batch_size, in_channels=input_channels, 
                        random_state=42, learning_rate=learning_rate)
    tcn.fit(X=datas_eeg, y=labels, n_epochs=100)
    # 下面开始进行随机搜索超参数
    # param_distribs = {
    #                   "num_channels":[[20]*11, [10]*11, [40]*11, [80]*11, [120]*11, [200]*11],
    #                   "batch_size":[2, 4, 8, 16], 
    #                   "learning_rate":[0.001, 0.0001, 0.0005], 
    #                   "dropout":[0.5, 0.8, 0.2, 0.1]}
    # rnd_search = RandomizedSearchCV(TCNClassifier(random_state=42, sequence_length=seq_length, 
    #                                               in_channels=input_channels, kernel_size=kernel_size), 
    #                                               param_distribs, n_iter=50, random_state=42, verbose=2)
    # rnd_search.fit(X=datas_eeg, y=labels, n_epochs=5)

