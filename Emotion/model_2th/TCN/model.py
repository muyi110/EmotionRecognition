#!/usr/bin/env python3
# -*- coding:UTF-8 -*-
import sys
sys.path.append("../common/") # 将其他模块路径添加到系统搜索路径
import numpy as np
import tensorflow as tf
import time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from tcn import TCN
from read_data import read_data, index_generator

tf.set_random_seed(42)
np.random.seed(42)

# 构建 TCN 模型类，为了兼容 scikit-learning 的 RandomizedSearchCV 类，后续可能实现超参数搜索
class TCNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 sequence_length, 
                 kernel_size,
                 num_channels=[30]*6, 
                 dropout=0.5, 
                 batch_size=16, 
                 in_channels=32, 
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
    def _TCN(self, inputs, n_outputs, training):
        '''构建 TCN 模型'''
        outputs = TCN(inputs, n_outputs, self.num_channels, 
                      self.sequence_length, self.kernel_size, self.dropout, is_training=training)
        return outputs
    def _bulid_graph(self, n_outputs):
        '''构建计算图'''
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)
        inputs = tf.placeholder(tf.float32, 
                                shape=(None, self.sequence_length, self.in_channels), name="inputs")
        labels = tf.placeholder(tf.int32, shape=(None), name="labels")
        self._training = tf.placeholder_with_default(False, shape=(), name="training") # 表示是训练阶段还是测试阶段
        learning_rate_ = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        tcn_outputs = self._TCN(inputs, n_outputs, self._training)
        predictions = tf.nn.softmax(tcn_outputs, name="predictions")
        # 计算交叉熵
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=tcn_outputs)
        loss = tf.reduce_mean(xentropy, name="loss")
        # 构建优化器节点
        optimizer = self.optimizer_class(learning_rate=learning_rate_)
        training_op = optimizer.minimize(loss)
        # 构建计算准确率节点
        correct = tf.nn.in_top_k(tcn_outputs, labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        # 构建全局初始化节点和模型保存节点
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self._X, self._y = inputs, labels
        self._learning_rate = learning_rate_
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
    def fit(self, X, y, n_epochs, X_valid=None, y_valid=None, X_test=None, y_test=None):
        '''Fit the model to the training set. If X_valid and y_valid are provided, use early stopping'''
        self.close_session()
        print("X test shape: ", X_test.shape)
        print("y test shape: ", y_test.shape)
        self.classes_ = np.unique(y)
        n_outputs = len(self.classes_) # 获取输出的类别数
        self.class_to_index_ = {label:index for index, label in enumerate(self.classes_)}
        y = np.array([self.class_to_index_[label] for label in y], dtype=np.int32)
        
        self.y_test_classes_ = np.unique(y_test)
        y_test_n_outputs = len(self.y_test_classes_) # 获取输出的类别数
        self.y_test_class_to_index_ = {label:index for index, label in enumerate(self.y_test_classes_)}
        y_test = np.array([self.y_test_class_to_index_[label] for label in y_test], dtype=np.int32)
        
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._bulid_graph(n_outputs) # 构建计算模型
        # 下面几个变量用于 early stopping
        max_check_without_progress = 20
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
                if epoch != 0 and epoch // 100 != 0:
                    self.learning_rate = 0.0002
                if epoch != 0 and epoch // 150 != 0:
                    self.learning_rate = 0.0001
                start_time = time.time()
                for X_batch_index, y_batch_index in index_generator(len(y), self.batch_size, seed=seed):
                    X_batch = X[X_batch_index]
                    y_batch = y[y_batch_index]
                    sess.run(self._training_op, feed_dict={self._X: X_batch, self._y: y_batch, self._training:True, self._learning_rate:self.learning_rate})

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
                          total_loss/(len(y)//8), (total_acc / (len(y)//8))*100, end_time-start_time))
                    if X_test is not None and y_test is not None and epoch % 1 == 0:
                        total_acc_test = 0
                        total_loss_test = 0
                        for i in range(len(y_test) // 8):
                            X_batch_test = X_test[i*8:(i+1)*8, :, :]
                            y_batch_test = y_test[i*8:(i+1)*8]
                            loss_test, acc_test = sess.run([self._loss, self._accuracy], 
                                                            feed_dict={self._X:X_batch_test, self._y:y_batch_test, self._training:False})
                            total_acc_test += acc_test
                            total_loss_test += loss_test
                        if total_acc_test >= best_acc:
                            best_acc = total_acc_test
                            self.save("./my_model/train_model.ckpt") # 将训练模型保存
                        print("learning rate: ", self.learning_rate)
                        print("Test accuracy: {:.4f}%\t  Test loss: {:.6f}".format((total_acc_test / (len(y_test) // 8))*100, total_loss_test/(len(y_test) // 8)))
                        # loss_test, acc_test = sess.run([self._loss, self._accuracy], 
                        #                                feed_dict={self._X:X_test, self._y:y_test})
                        # print("Test accuracy: {:.4f}%\t Test loss: {:.6f}".format(acc_test*100, loss_test))
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
        return np.array([self.classes_[class_index] for class_index in class_indices], np.int32).reshape(-1)
    def save(self, path):
        self._saver.save(self._session, path)
    def restore(self, path="./my_model/train_model.ckpt"):
        self._saver.restore(self._session, path)

if __name__ == "__main__":
    # 开始将数据集划分为训练集和测试集
    np.random.seed(42)
    permutation = list(np.random.permutation(40)) # 将数据随机打乱
    train_index = [1]  # 选择某一个人
    test_index = [1]   # 选择某一个人
    trials_list = []
    train_list = list(permutation[8:40])
    test_list = list(permutation[0:8])
    temp1 = (train_list, test_list)
    trials_list.append(temp1)
    train_list = list(permutation[0:8]) + list(permutation[16:40])
    test_list = list(permutation[8:16])
    temp1 = (train_list, test_list)
    trials_list.append(temp1)
    train_list = list(permutation[0:16]) + list(permutation[24:40])
    test_list = list(permutation[16:24])
    temp1 = (train_list, test_list)
    trials_list.append(temp1)
    train_list = list(permutation[0:24]) + list(permutation[32:40])
    test_list = list(permutation[24:32])
    temp1 = (train_list, test_list)
    trials_list.append(temp1)
    train_list = list(permutation[0:32])
    test_list = list(permutation[32:40])
    temp1 = (train_list, test_list)
    trials_list.append(temp1)
    assert(len(trials_list) == 5)
    num_ = 0
    F1_scores_list = []
    accuracy_list = []
    samples_info = []
    for train_trial_list, test_trial_list in trials_list:
        num_ = num_ + 1
        # 获取生理信号数据
        # datas_train, train_labels = read_data(people_list=train_index, classify_object_name=0, train_flag=True,
        #                                       trial_list=train_trial_list, windows=9, overlapping=8, 
        #                                       cross_validation_number=num_)
        # datas_test, test_labels = read_data(people_list=test_index, classify_object_name=0, train_flag=False, 
        #                                     trial_list=test_trial_list, windows=9, overlapping=8, 
        #                                     cross_validation_number=num_)
        datas_train = np.load("../common/samples_single_people/valence_old/s0/train_datas"+str(num_)+".npy") 
        train_labels = np.load("../common/samples_single_people/valence_old/s0/train_labels"+str(num_)+".npy") 
        datas_test = np.load("../common/samples_single_people/valence_old/s0/test_datas"+str(num_)+".npy")  
        test_labels = np.load("../common/samples_single_people/valence_old/s0/test_labels"+str(num_)+".npy")  
        datas_train = np.array(datas_train)
        train_labels = np.array(train_labels)
        datas_test = np.array(datas_test)
        test_labels = np.array(test_labels)
        print("train data set number: ", len(train_labels))
        print("train datas shape: ", datas_train.shape)
        print("test data set number: ", len(test_labels))
        print("test datas shape: ", datas_test)
        print("train label 0: ", sum(train_labels==0), " train label 1: ", sum(train_labels==1))
        print("test label 0: ", sum(test_labels==0), " test label 1: ", sum(test_labels==1))
        train_label_0 = sum(train_labels==0)
        test_label_0 = sum(test_labels==0)
        label_0 = (train_label_0, test_label_0)
        samples_info.append(label_0)

        datas_train = datas_train.transpose((0,2,1))
        datas_test = datas_test.transpose((0,2,1))
        # datas_train = datas_train.reshape(datas_train.shape[0], -1, 1)
        # datas_test = datas_test.reshape(datas_test.shape[0], -1, 1)
        print("train number: ", len(train_labels))
        print(datas_train.shape, train_labels.shape)
        print("test number: ", len(test_labels))
        print(datas_test.shape, test_labels.shape)

        n_classes = 2 # 貌似没有用到------------
        input_channels = datas_train.shape[-1]
        seq_length = datas_train.shape[-2] # 序列的长度
        dropout = 0.5
        learning_rate=0.001
        num_channels = [128, 64, 32] # 有多少层，及每一层包含的神经元个数（这里的一层指一个 block）
        kernel_size = 3   # 卷积核大小  
        batch_size = 64

        # 开始构建TCN 模型实例
        tcn = TCNClassifier(num_channels=num_channels, sequence_length = seq_length, kernel_size=kernel_size, 
                            dropout=dropout, batch_size=batch_size, in_channels=input_channels, 
                            random_state=42, learning_rate=learning_rate)
        tcn.fit(X=datas_train, y=train_labels, n_epochs=351, X_test=datas_test, y_test=test_labels)
        tcn.restore()
        total_acc_test = 0
        y_pred_labels = []
        for i in range(len(test_labels) // 8):
            X_batch_test = datas_test[i*8:(i+1)*8, :, :]
            y_batch_test = test_labels[i*8:(i+1)*8]
            y_pred = tcn.predict(X_batch_test)
            y_pred_labels += list(y_pred)
            total_acc_test += accuracy_score(y_batch_test, y_pred)
        print("Test accuracy: {:.4f}%".format((total_acc_test / (len(test_labels) // 8))*100))
        F1_scores_list.append(f1_score(test_labels, np.array(y_pred_labels)))
        total_acc_test1 = 0
        total_loss_test = 0
        for i in range(len(test_labels) // 8):
            X_batch_test = datas_test[i*8:(i+1)*8, :, :]
            y_batch_test = test_labels[i*8:(i+1)*8]
            loss_test, acc_test = tcn._session.run([tcn._loss, tcn._accuracy], 
                                            feed_dict={tcn._X:X_batch_test, tcn._y:y_batch_test})
            total_acc_test1 += acc_test
            total_loss_test += loss_test
        print("Test accuracy: {:.4f}%\t  Test loss: {:.6f}".format((total_acc_test1 / (len(test_labels) // 8))*100, total_loss_test/(len(test_labels) // 8)))
        temp = (total_acc_test / (len(test_labels)//8), total_acc_test1 / (len(test_labels)//8))
        accuracy_list.append(temp)
    print("-------------------------------accuracy_list--------------------------------------")
    print(accuracy_list)
    print("-------------------------------F1_score--------------------------------------")
    print(F1_scores_list)
    print("-------------------------------sampels info--------------------------------------")
    print(samples_info)
