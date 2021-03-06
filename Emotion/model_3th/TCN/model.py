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
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_mutual_info_score
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
        outputs, self.feature = TCN(inputs, n_outputs, self.num_channels, 
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
    def fit(self, X, y, n_epochs, X_valid=None, y_valid=None, X_test=None, y_test=None, people_num_=None):
        '''Fit the model to the training set. If X_valid and y_valid are provided, use early stopping'''
        if people_num_ is None:
            self._num_ = ""
        elif people_num_ is not None:
            self._num_ = str(people_num_)
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
                    loss_train, acc_train = sess.run([self._loss, self._accuracy], 
                                                  feed_dict={self._X:X_batch, self._y:y_batch})
                    total_loss = loss_train
                    total_acc = acc_train
                    end_time = time.time()
                    print("{} training loss: {:.6f}\t|  training accuracy: {:.2f}% | time: {:.2f}s".format(epoch, 
                          total_loss, (total_acc)*100, end_time-start_time))
                    if X_test is not None and y_test is not None and epoch % 1 == 0:
                        total_acc_test = 0
                        total_loss_test = 0
                        loss_test, acc_test = sess.run([self._loss, self._accuracy], 
                                                        feed_dict={self._X:X_test, self._y:y_test, self._training:False})
                        total_acc_test = acc_test
                        total_loss_test = loss_test
                        if total_acc_test >= best_acc:
                            best_acc = total_acc_test
                            self.save("./my_model/"+self._num_+"/train_model_A.ckpt") # 将训练模型保存
                        print("Test accuracy: {:.4f}%\t  Test loss: {:.6f}".format((total_acc_test )*100, total_loss_test))
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
    def restore(self):
        path="./my_model/"+self._num_+"/train_model_A.ckpt"
        self._saver.restore(self._session, path)

def split_datas_with_cross_validation(datas, labels, windows, seed=None):
    samples_nums_one_trial = 60 - windows + 1
    assert(datas.shape == ((60-windows+1)*40, windows, 128))
    assert(labels.shape == ((60-windows+1)*40,))
    assert(sum(labels[:samples_nums_one_trial]) == 0 or sum(labels[:samples_nums_one_trial]) == samples_nums_one_trial)
    datas_one = datas[list(labels==0)]
    datas_two = datas[list(labels==1)]
    labels_one = labels[list(labels==0)]
    labels_two = labels[list(labels==1)]
    assert(len(datas_one) // samples_nums_one_trial + len(datas_two) // samples_nums_one_trial == 40)

    label_one_number = (labels == 0).sum() # 获取类别 1 的数目
    label_two_number = (labels == 1).sum() # 获取类别 2 的数目   
    trial_label_one_number = int(label_one_number / samples_nums_one_trial) # 属于类别 1 的实验数
    trial_label_two_number = int(label_two_number / samples_nums_one_trial) # 属于类别 2 的实验数
    # 5 折交叉验证， 训练集有 32 个实验， 测试集有 8 个实验
    label_one_train_number = int(round(trial_label_one_number * 0.8))
    label_two_train_number = int(round(trial_label_two_number * 0.8))
    assert(label_one_train_number + label_two_train_number == 32)
    label_one_test_number = trial_label_one_number - label_one_train_number
    label_two_test_number = trial_label_two_number - label_two_train_number
    assert(label_one_test_number + label_two_test_number == 8)
    np.random.seed(seed)
    permutation_one = list(np.random.permutation(trial_label_one_number)) # 将数据随机打乱
    permutation_two = list(np.random.permutation(trial_label_two_number)) # 将数据随机打乱
    train_one_index = permutation_one[:label_one_train_number]
    train_two_index = permutation_two[:label_two_train_number]
    test_one_index = permutation_one[label_one_train_number:]
    test_two_index = permutation_two[label_two_train_number:]
    # 获取训练集和测试集
    train_datas = []
    train_labels = []
    for elem in train_one_index:
        train_datas.append(datas_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        train_labels.append(labels_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial].reshape(1,-1))
    for elem in train_two_index:
        train_datas.append(datas_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        train_labels.append(labels_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial].reshape(1,-1))
    datas_train = np.r_[tuple(train_datas)]
    labels_train = np.c_[tuple(train_labels)].reshape(-1)
    assert(datas_train.shape == (32*samples_nums_one_trial, windows, 128))
    assert(labels_train.shape == (32*samples_nums_one_trial,))
    test_datas = []
    test_labels = []
    for elem in test_one_index:
        test_datas.append(datas_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        test_labels.append(labels_one[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial].reshape(1,-1))
    for elem in test_two_index:
        test_datas.append(datas_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial])
        test_labels.append(labels_two[elem*samples_nums_one_trial:(elem+1)*samples_nums_one_trial].reshape(1,-1))
    datas_test = np.r_[tuple(test_datas)]
    labels_test = np.c_[tuple(test_labels)].reshape(-1)
    assert(datas_test.shape == (8*samples_nums_one_trial, windows, 128))
    assert(labels_test.shape == (8*samples_nums_one_trial,))
    return datas_train, labels_train, datas_test, labels_test

def split_datas_with_cross_validatioan_random(datas, labels, windows, seed=None):
    samples_nums_one_trial = 60 - windows + 1
    assert(datas.shape == ((60-windows+1)*40, windows, 128))
    assert(labels.shape == ((60-windows+1)*40,))
    assert(sum(labels[:samples_nums_one_trial]) == 0 or sum(labels[:samples_nums_one_trial]) == samples_nums_one_trial)
    np.random.seed(seed)
    permutation = list(np.random.permutation(datas.shape[0]))
    permutation_train = permutation[:int(labels.shape[0]*0.8)]
    permutation_test = permutation[int(labels.shape[0]*0.8):]
    datas_train = datas[permutation_train]
    labels_train = labels[permutation_train]
    datas_test = datas[permutation_test]
    labels_test = labels[permutation_test]
    return datas_train, labels_train, datas_test, labels_test

if __name__ == "__main__":
    people_num_list = list(range(0, 32))
    windows = 9 # 样本窗口大小
    accuracy_results_dic = {} # 一个字典，保存最终的结果
    F1_score_results_dic = {} # 一个字典，保存最终的结果
    samples_info_dic = {}
    silhouette_score_dic = {}
    normalized_mutual_info_score_dic = {}
    adjusted_mutual_info_score_dic = {}
    for people_num_ in people_num_list:
        datas, labels = read_data(people_list=[people_num_], windows=windows, overlapping=windows-1, 
                                  classify_object_name=1, mv_flag=True, lds_flag=False)
        datas = np.array(datas)
        labels = np.array(labels)
        datas = datas.transpose((0,2,1))
        assert(datas.shape == ((60-windows+1)*40, windows, 128))
        assert(labels.shape == ((60-windows+1)*40,))
        seed_ = 1
        cross_validation_number = 5
        F1_scores_list = []
        accuracy_list = []
        samples_info = []
        silhouette_score_list = []
        normalized_mutual_info_score_list = []
        adjusted_mutual_info_score_list = []
        for number in range(cross_validation_number):
            seed_ = seed_ + 1
            datas_train, train_labels, datas_test, test_labels = split_datas_with_cross_validation(datas, 
                                                                 labels, windows, seed_)
            # datas_train, train_labels, datas_test, test_labels = split_datas_with_cross_validatioan_random(datas, 
            #                                                      labels, windows, seed_)
            print("train label 0: ", sum(train_labels==0), " train label 1: ", sum(train_labels==1))
            print("test label 0: ", sum(test_labels==0), " test label 1: ", sum(test_labels==1))
            train_label_0 = sum(train_labels==0)
            test_label_0 = sum(test_labels==0)
            label_0 = (train_label_0, test_label_0)
            samples_info.append(label_0)

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
            tcn.fit(X=datas_train, y=train_labels, n_epochs=351, X_test=datas_test, y_test=test_labels,
                    people_num_=windows)
            tcn.restore()
            y_pred = tcn.predict(datas_test)
            total_acc_test = accuracy_score(test_labels, y_pred)
            y_pred_labels = list(y_pred)
            print("Test accuracy: {:.4f}%".format(total_acc_test*100))

            F1_scores_list.append(f1_score(test_labels, np.array(y_pred_labels)))
            temp = total_acc_test
            accuracy_list.append(temp)
            # 下面计算类别结果评价指标
            normalized_mutual_info_score_list.append(normalized_mutual_info_score(test_labels, y_pred.reshape(-1)))
            adjusted_mutual_info_score_list.append(adjusted_mutual_info_score(test_labels, y_pred.reshape(-1)))
            if(len(np.unique(y_pred.reshape(-1))) == 2):
                silhouette_score_list.append(silhouette_score(tcn._session.run(tcn.feature, feed_dict={tcn._X:datas_test}), y_pred.reshape(-1)))
            else:
                silhouette_score_list.append(0)

            tf.reset_default_graph()
        print("-------------------------------accuracy_list--------------------------------------")
        print(accuracy_list)
        print("accuacy mean : ", sum(accuracy_list) / 5)
        print("accuacy min: ", min(accuracy_list))
        print("accuacy max: ", max(accuracy_list))
        print("-------------------------------F1 score--------------------------------------")
        print(F1_scores_list)
        print("F1 score mean: ",sum(F1_scores_list)/5)
        print("F1 score min: ", min(F1_scores_list))
        print("F1 score max: ", max(F1_scores_list))
        print("-------------------------------sampels info--------------------------------------")
        print(samples_info)

        print("-------------------------------metrics--------------------------------------")
        print("adjusted_mutual_info_score: ", adjusted_mutual_info_score_list)
        print("adjusted_mutual_info_score mean: ", sum(adjusted_mutual_info_score_list) / 5)
        print("normalized_mutual_info_score: ", normalized_mutual_info_score_list)
        print("normalized_mutual_info_score mean: ", sum(normalized_mutual_info_score_list) / 5)
        print("silhouette_score: ", silhouette_score_list)
        print("silhouette_score mean: ", sum(silhouette_score_list) / 5)
        accuracy_results_dic[str(people_num_)] = accuracy_list + \
                                              [min(accuracy_list), max(accuracy_list), sum(accuracy_list)/5]
        F1_score_results_dic[str(people_num_)] = F1_scores_list + \
                                              [min(F1_scores_list), max(F1_scores_list), sum(F1_scores_list)/5]
        samples_info_dic[str(people_num_)] = samples_info
        adjusted_mutual_info_score_dic[str(people_num_)] = adjusted_mutual_info_score_list + [sum(adjusted_mutual_info_score_list) / 5]
        normalized_mutual_info_score_dic[str(people_num_)] = normalized_mutual_info_score_list + [sum(normalized_mutual_info_score_list) / 5]
        silhouette_score_dic[str(people_num_)] = silhouette_score_list + [sum(silhouette_score_list) / 5]
        print("accuracy: ")
        print(accuracy_results_dic)
        print("F1 score: ")
        print(F1_score_results_dic)
        print("adjusted_mutual_info_score: ")
        print(adjusted_mutual_info_score_dic)
        print("normalized_mutual_info_score:")
        print(normalized_mutual_info_score_dic)
        print("silhouette_score")
        print(silhouette_score_dic)
    np.save("./result/mv_new/arousal/"+str(windows)+"/accuracy_0804_128", accuracy_results_dic)
    np.save("./result/mv_new/arousal/"+str(windows)+"/F1_score_0804_128", F1_score_results_dic)
    np.save("./result/mv_new/arousal/"+str(windows)+"/samples_0804_128", samples_info_dic)
    np.save("./result/mv/arousal/"+str(windows)+"/adjusted_mutual_info_score_0804_128", adjusted_mutual_info_score_dic)
    np.save("./result/mv/arousal/"+str(windows)+"/normalized_mutual_info_score_dic_0804_128", normalized_mutual_info_score_dic)
    np.save("./result/mv/arousal/"+str(windows)+"/silhouette_score_dic_0804_128", silhouette_score_dic)
    print("accuracy: ")
    print(accuracy_results_dic)
    sum_ = 0
    for i in range(32):
        sum_ += accuracy_results_dic[str(i)][-1]
    print("acc: ", sum_/32)
    print("F1 score: ")
    print(F1_score_results_dic)
    sum_ = 0
    for i in range(32):
        sum_ += F1_score_results_dic[str(i)][-1]
    print("f1 acc: ", sum_/32)
    print("sample info")
    print(samples_info_dic)
    print("metrics: ")
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    for i in range(32):
        sum_1 += adjusted_mutual_info_score_dic[str(i)][-1]
        sum_2 += normalized_mutual_info_score_dic[str(i)][-1]
        sum_3 += silhouette_score_dic[str(i)][-1]
    print("adjusted_mutual_info_score: ", sum_1 / 32)
    print("normalized_mutual_info_score: ", sum_2 / 32)
    print("silhouette_score: ", sum_3 / 32)
