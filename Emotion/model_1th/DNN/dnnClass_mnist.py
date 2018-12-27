#!/usr/bin/env python3
#! -*- coding:UTF-8 -*-
import time
import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from data import read_data, index_generator

tf.set_random_seed(42)
np.random.seed(42)
# 创建 He initializer 节点，适用于 ReLU 及其变体激活函数
he_init = tf.contrib.layers.variance_scaling_initializer()
#创建 DNNClassifier 类，为了兼容 scikit-learn 的 RandomizedSearchCV 类，实现超参数搜索调整
class DNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden_layers=5, 
                 n_neurons=100,
                 optimizer_class=tf.train.AdamOptimizer,
                 learning_rate=0.001, 
                 batch_size=32, 
                 activation=tf.nn.relu, 
                 initializer=he_init, 
                 batch_norm_momentum=None, 
                 dropout_rate=None, 
                 random_state=None
                 ):
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.initializer = initializer
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self._session = None
    def _dnn(self, inputs):
        '''构建隐含层，支持 batch normalization 和 dropout'''
        #每一层的 batch normalization 和 dropout 都一样
        for layer in range(self.n_hidden_layers):
            if self.dropout_rate:
                inputs = tf.layers.dropout(inputs, self.dropout_rate, training=self._training)
            # tf.layers.dense 默认没有激活函数，输出是线性的
            inputs = tf.layers.dense(inputs, self.n_neurons, 
                                     kernel_initializer=self.initializer,
                                     name="hidden%d" % (layer+1))
            if self.batch_norm_momentum:
                inputs = tf.layers.batch_normalization(inputs, momentum=self.batch_norm_momentum, 
                                                      training=self._training)
            inputs = self.activation(inputs, name="hidden%d_out" % (layer+1))
        return inputs
    def _build_graph(self, n_inputs, n_outputs):
        '''构建模型'''
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.int32, shape=(None), name="y")
        # 如果需要 BN 或者 dropout 需要指示是在训练阶段还是其他阶段
        if self.batch_norm_momentum or self.dropout_rate:
            self._training = tf.placeholder_with_default(False, shape=(), name="training")
        else:
            self._training=None
        dnn_outputs = self._dnn(X) # 构建模型，获取最后隐藏层的输出
        logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=he_init, name="logits")
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
    def fit(self, X, y, n_epochs=100, X_valid=None, y_valid=None, X_test=None, y_test=None):
        '''Fit the model to the training set. If X_valid and y_valid are provided, use early stopping'''
        self.close_session()
        # 获取 n_inputs 和 n_outputs
        n_inputs = X.shape[1]  # 可以理解为获取特征数
        self.classes_ = np.unique(y) # 获取有几个类别, 返回是一个有序的对象（升序）
        n_outputs = len(self.classes_)
        # 将 labels vector 转为类索引。例如：y=[8,8,9,5,7,6,6,6] 得到有序的类标签是 [5,6,7,8,9] 
        # label vector 将会转化为 [3,3,4,0,2,1,1,1]
        self.class_to_index_ = {label: index for index, label in enumerate(self.classes_)}
        y = np.array([self.class_to_index_[label] for label in y], dtype=np.int32)
        
        self.y_test_classes_ = np.unique(y_test) # 获取有几个类别, 返回是一个有序的对象（升序）
        y_test_n_outputs = len(self.y_test_classes_)
        self.y_test_class_to_index_ = {label: index for index, label in enumerate(self.y_test_classes_)}
        y_test = np.array([self.y_test_class_to_index_[label] for label in y_test], dtype=np.int32)
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs) # 调用构建模型方法
        # 下面的几个变量用于 early stopping
        max_checks_without_progress = 20
        checks_without_progress = 0
        best_loss = np.infty
        best_params = None
        # 开始训练阶段
        best_acc = 0 # 测试集最好的准确率
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
                    sess.run(self._training_op, feed_dict={self._X: X_batch, self._y: y_batch})
                # X_valid 和 y_valid 验证集用于早停算法
                if X_valid is not None and y_valid is not None:
                    loss_val, acc_val = sess.run([self._loss, self._accuracy], 
                                                 feed_dict={self._X: X_valid, self._y: y_valid})
                    if loss_val < best_loss:
                        best_params = self._get_model_params()
                        best_loss = loss_val
                        checks_without_progress = 0
                    else:
                        checks_without_progress += 1
                    print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(epoch, 
                          loss_val, best_loss, acc_val*100))
                    if checks_without_progress >= max_checks_without_progress:
                        print("Early stopping")
                        break
                else:
                    total_loss = 0 
                    total_acc = 0
                    for i in range(len(y) // 8):
                        X_batch = X[i*8:(i+1)*8,:]
                        y_batch = y[i*8:(i+1)*8]
                        loss_train, acc_train = sess.run([self._loss, self._accuracy], 
                                                      feed_dict={self._X:X_batch, self._y:y_batch})
                        total_loss += loss_train
                        total_acc += acc_train
                        end_time = time.time()
                    print("{}\ttraining loss: {:.6f}\t|  training accuracy: {:.2f}% | time: {:.2f}s".format(epoch, 
                          total_loss/(len(y)//8), (total_acc / (len(y)//8))*100, end_time-start_time))
                    if X_test is not None and y_test is not None and epoch % 1 == 0:
                        total_acc_test = 0
                        total_loss_test = 0
                        for i in range(len(y_test) // 8):
                            X_batch_test = X_test[i*8:(i+1)*8, :]
                            y_batch_test = y_test[i*8:(i+1)*8]
                            loss_test, acc_test = sess.run([self._loss, self._accuracy], 
                                                            feed_dict={self._X:X_batch_test, self._y:y_batch_test})
                            total_acc_test += acc_test
                            total_loss_test += loss_test
                        if total_acc_test > best_acc:
                            best_acc = total_acc_test
                            self.save("./my_model/train_model.ckpt") # 将训练模型保存
                        print("Test accuracy: {:.4f}%\t  Test loss: {:.6f}".format((total_acc_test / (len(y_test) // 8))*100, total_loss_test/(len(y_test) // 8)))
            if best_params:
                self._restore_model_params(best_params)
            return self
    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._Y_proba.eval(feed_dict={self._X: X})
    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([[self.classes_[class_index]] for class_index in class_indices], np.int32)
    def save(self, path):
        self._saver.save(self._session, path)
    def restore(self, path="./my_model/train_model.ckpt"):
        self._saver.restore(self._session, path)
if __name__=="__main__":
    datas, labels = read_data(windows=1, overlapping=0, raw_data=True)
    datas = np.array(datas)
    labels = np.array(labels)
    print("data set number: ", len(labels))
    print("datas shape: ", datas.shape)
    # 开始将数据集划分为训练集和测试集
    np.random.seed(42)
    permutation = list(np.random.permutation(len(labels))) # 将数据随机打乱
    train_index = permutation[:-int(len(labels)*0.3)]
    test_index = permutation[-int(len(labels)*0.3):]
    datas_train = datas[train_index]
    train_labels = labels[train_index]
    datas_test = datas[test_index]
    test_labels = labels[test_index]
    del datas # 释放内存
    datas_train = datas_train.transpose((0,2,1))
    datas_test = datas_test.transpose((0,2,1))
    datas_train = datas_train.reshape(datas_train.shape[0], -1) # not raw data
    datas_test = datas_test.reshape(datas_test.shape[0], -1) # not raw data
    print("train number: ", len(train_labels))
    print(datas_train.shape, train_labels.shape)
    print("test number: ", len(test_labels))
    print(datas_test.shape, test_labels.shape)
    dnn_clf_bn = DNNClassifier(learning_rate=0.0001, n_neurons=100, batch_size=32, random_state=42)
    dnn_clf_bn.fit(datas_train, train_labels, n_epochs=101, X_valid=None, y_valid=None, X_test=datas_test, y_test=test_labels)
    dnn_clf_bn.restore()
    total_acc_test = 0
    for i in range(len(test_labels) // 8):
        X_batch_test = datas_test[i*8:(i+1)*8, :]
        y_batch_test = test_labels[i*8:(i+1)*8]
        y_pred = dnn_clf_bn.predict(X_batch_test)
        total_acc_test += accuracy_score(y_batch_test, y_pred)
    print("Test accuracy: {:.4f}%".format((total_acc_test / (len(test_labels) // 8))*100))
