# -*- coding:UTF-8 -*-
################################################################
# 此部分实现 DANN 网络的构建，特征提取基于 TCN 网络
################################################################
import tensorflow as tf
import numpy as np
from tcn import TCN
from sklearn.exceptions import NotFittedError
from gradient_reversal import gradient_reversal
from utils import batch_generator

class MDANNModel():
    def __init__(self, 
                 sequence_length, 
                 kernel_size, 
                 num_channels, 
                 dropout,
                 batch_size,
                 in_channels,
                 train_ratio=0.5,
                 random_state=None, 
                 optimizer_class=tf.train.MomentumOptimizer):
        self.sequence_length = sequence_length
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.dropout = dropout
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.train_ratio = train_ratio
        self.random_state = random_state
        self.optimizer_class = optimizer_class
        self._session = None

    def _build_model(self, n_outputs):
        '''
        n_outputs: 输出的类别数
        '''
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.sequence_length, self.in_channels), name="inputs")
        self.labels = tf.placeholder(tf.int32, shape=(None), name="labels")
        self.domain_labels = tf.placeholder(tf.int32, shape=(None), name="domain_labels")
        self._training = tf.placeholder_with_default(False, shape=(), name="training")
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        self.num = tf.placeholder(tf.int32, shape=(), name="num")
        self.l = tf.placeholder(tf.float32, shape=(), name="l") # 用于梯度翻转层常数
        # TCN 模型用于提取特征
        with tf.variable_scope("feature_extractor"):
            # feature.shape = (samples, seq_length, features)
            features = TCN(self.inputs, n_outputs, self.num_channels, self.sequence_length, 
                           self.kernel_size, self.dropout, is_training=self._training)
            # 取最后一个输出作为最终特征
            self.feature = features[:, -1, :]
            # W = tf.get_variable(name="W", dtype=tf.float32, shape=(9, 1, 1, 1), 
            #                     initializer=tf.contrib.layers.xavier_initializer())
            # features_temp = tf.nn.conv2d(tf.reshape(features, [-1, features.get_shape()[1], features.get_shape()[2], 1]), 
            #                              filter=W, padding="VALID", strides=[1, 1, 1, 1])
            # b = tf.get_variable('b', shape=[1], dtype=tf.float32, 
            #                     initializer=tf.zeros_initializer(), trainable=True)
            # features_temp = tf.nn.bias_add(features_temp, b)
            # features_temp = tf.sigmoid(features_temp)
            # print("fes: ", features_temp.get_shape())
            # self.feature = tf.layers.flatten(features_temp)
            # print("fes: ", self.feature.get_shape())
            # # 特征提取尝试 RNN 网络
            # lstm_cells = tf.nn.rnn_cell.LSTMCell(num_units=120)
            # outputs, states = tf.nn.dynamic_rnn(lstm_cells, self.inputs, dtype=tf.float32)
            # self.feature = outputs[:, -1, :]
            # print("fes: ", self.feature.get_shape())
        # softmax 用于分类预测
        with tf.variable_scope("label_predictor"):
            # all_features = lambda: self.feature
            # source_features = lambda: tf.slice(self.feature, [0, 0], [int(self.batch_size*self.train_ratio), -1])
            # classify_features = tf.cond(self._training, source_features, all_features)

            # all_labels = lambda: self.labels
            # source_labels = lambda: tf.slice(self.labels, [0], [int(self.batch_size*self.train_ratio)])
            # self.classify_labels = tf.cond(self._training, source_labels, all_labels)

            logits = tf.contrib.layers.fully_connected(self.feature, n_outputs, activation_fn=None) 
            self.predictor = tf.nn.softmax(logits, name="predictor")
            self.predictor_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, 
                                                                                 logits=logits)
        # softmax 用于域判别器
        with tf.variable_scope("domain_predictor_class_one"):
            self.feature_class_one = self.feature[:self.num]
            self.domain_labels_one = self.domain_labels[:self.num]
            # 反向传播时候翻转梯度
            feat_one = gradient_reversal(self.feature_class_one, self.l)
            #d_temp_one = tf.contrib.layers.fully_connected(feat_one, 32, activation_fn=tf.nn.relu, scope="d_temp")
            d_logits_one = tf.contrib.layers.fully_connected(feat_one, 14, activation_fn=None, scope="d_logits")
            self.domain_predictor_one = tf.nn.softmax(d_logits_one, name="domain_pre")
            self.domain_loss_one = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=d_logits_one, 
                                                                              labels=self.domain_labels_one)
        
        # softmax 用于域判别器
        with tf.variable_scope("domain_predictor_class_two"):
            self.feature_class_two = self.feature[self.num:]
            self.domain_labels_two = self.domain_labels[self.num:] - tf.cast(self.num / 4, tf.int32)
            # 反向传播时候翻转梯度
            feat_two = gradient_reversal(self.feature_class_two, self.l)
            #d_temp_two = tf.contrib.layers.fully_connected(feat_two, 32, activation_fn=tf.nn.relu, scope="d_temp")
            d_logits_two = tf.contrib.layers.fully_connected(feat_two, 18, activation_fn=None, scope="d_logits")
            self.domain_predictor_two = tf.nn.softmax(d_logits_two, name="domain_pre")
            self.domain_loss_two = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=d_logits_two, 
                                                                                  labels=self.domain_labels_two)

    def _build_graph(self, n_outputs):
        self._build_model(n_outputs)
        # 构建损失节点
        pred_loss = tf.reduce_mean(self.predictor_loss)
        domain_loss_one = tf.reduce_mean(self.domain_loss_one)
        domain_loss_two = tf.reduce_mean(self.domain_loss_two)
        total_loss = pred_loss + domain_loss_one + domain_loss_two
        domain_loss = domain_loss_one + domain_loss_two
        # 构建训练节点
        pre_train_op = self.optimizer_class(self.learning_rate, 0.9).minimize(pred_loss)
        dann_train_op = self.optimizer_class(self.learning_rate, 0.9).minimize(total_loss)
        self.domain_train_op = self.optimizer_class(self.learning_rate, 0.9).minimize(domain_loss)
        # 构建评估节点
        correct_label_predictor = tf.nn.in_top_k(self.predictor, self.labels, 1)
        label_acc = tf.reduce_mean(tf.cast(correct_label_predictor, tf.float32))
        correct_domain_predictor_one = tf.nn.in_top_k(self.domain_predictor_one, self.domain_labels_one, 1)
        domain_acc_one = tf.reduce_mean(tf.cast(correct_domain_predictor_one, tf.float32))
        correct_domain_predictor_two = tf.nn.in_top_k(self.domain_predictor_two, self.domain_labels_two, 1)
        domain_acc_two = tf.reduce_mean(tf.cast(correct_domain_predictor_two, tf.float32))
        # 构建全局初始化节点
        init = tf.global_variables_initializer()
        # 构建模型保存节点
        saver = tf.train.Saver()

        self._pred_loss, self._domain_loss_one, self._domain_loss_two = pred_loss, domain_loss_one, domain_loss_two
        self._regular_train_op, self._dann_train_op = pre_train_op, dann_train_op
        self._label_acc, self._domain_acc_one, self._domain_acc_two = label_acc, domain_acc_one, domain_acc_two
        self._init, self._saver = init, saver

    def close_session(self):
        if self._session:
            self._session.close()

    def fit(self, X, y, num_steps, X_test, y_test, outputs, training_mode="dann", people_num=None):
        if people_num is None:
            self._num_ = ""
        elif people_num is not None:
            self._num_ = str(people_num)
        self.close_session()
        self.classes_ = np.unique(y)
        n_outputs = len(self.classes_) # 获取输出的类别数
        assert(n_outputs == outputs)
        self.class_to_index_ = {label:index for index, label in enumerate(self.classes_)}
        y = np.array([self.class_to_index_[label] for label in y], dtype=np.int32)
        
        self.y_test_classes_ = np.unique(y_test)
        self.y_test_class_to_index_ = {label:index for index, label in enumerate(self.y_test_classes_)}
        y_test = np.array([self.y_test_class_to_index_[label] for label in y_test], dtype=np.int32)
        
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(n_outputs)
        # 开始训练阶段
        best_acc = 0 # 测试集最好的准确率
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            sess.run(self._init)
            # batch 生成器
            gen_batch = batch_generator([X, y], self.batch_size)
            gen_source_only_batch = batch_generator([X, y], self.batch_size)
            temp_list = [] # 存放域标签的临时列表
            for i in range(32): # 一共有 32 的域
                temp_list += [i]*(self.batch_size//32)
            domain_label = np.array(temp_list).reshape(-1)
            for step in range(num_steps):
                p = float(step) / num_steps
                l = 2. / (1. + np.exp(-10*p)) - 1
                l *= 10
                lr = 0.001 / (1 + 10 * p)**0.75
                
                if training_mode == "dann":
                    X_batch, y_batch, num_0 = next(gen_batch)
                    assert(num_0 // 4 == 14)
                    assert(y_batch.shape==(self.batch_size,))
                    assert(X_batch.shape==(self.batch_size, self.sequence_length, self.in_channels))
                    if step % 4 == 0:
                       sess.run(self.domain_train_op, feed_dict={self.inputs:X_batch, self.labels:y_batch, 
                                                                self.domain_labels:domain_label, self._training:True, 
                                                                self.l:l, self.learning_rate:lr, self.num:num_0})
                    sess.run(self._regular_train_op, feed_dict={self.inputs:X_batch, self.labels:y_batch, 
                                                             self.domain_labels:domain_label, self._training:True, 
                                                             self.l:l, self.learning_rate:lr, self.num:num_0})
                    # print info
                    if step % 50 == 0 and step != 0:
                        train_pred_total_loss, train_pred_acc = sess.run([self._pred_loss, self._label_acc], 
                                                                          feed_dict={self.inputs:X, self.labels:y})
                        # domain_loss, domain_acc = sess.run([self._domain_loss, self._domain_acc], 
                        #                                    feed_dict={self.inputs:X_batch, self.domain_labels:domain_label})
                        print("{}\ttraining loss: {:.4f}\t|  trainning accuracy: {:.2f}%".format(step, 
                              train_pred_total_loss, train_pred_acc*100))
                        # print("domain loss: {:.2f}\t|  domain accuracy: {:.2f}%".format(domain_loss, domain_acc*100))
                        test_pred_total_loss, test_pred_acc = sess.run([self._pred_loss, self._label_acc], 
                                                                        feed_dict={self.inputs:X_test, self.labels:y_test})
                        if test_pred_acc >= best_acc:
                            best_acc = test_pred_acc
                            self.save("./my_model/"+self._num_+"/train_model.ckpt")
                        print("test loss: {:.4f}\t|  test accuracy: {:.2f}%".format(test_pred_total_loss, test_pred_acc*100))
                if training_mode == "source":
                    X_batch, y_batch, _ = next(gen_source_only_batch)
                    sess.run(self._regular_train_op, feed_dict={self.inputs:X_batch, self.labels:y_batch, 
                                                                self.learning_rate:lr})
                    # print info
                    if step % 50 == 0 and step != 0:
                        train_pred_total_loss, train_pred_acc = sess.run([self._pred_loss, self._label_acc], 
                                                                          feed_dict={self.inputs:X, self.labels:y})
                        print("{}\ttraining loss: {:.4f}\t|  trainning accuracy: {:.2f}%".format(step, 
                              train_pred_total_loss, train_pred_acc*100))
                        test_pred_total_loss, test_pred_acc = sess.run([self._pred_loss, self._label_acc], 
                                                                       feed_dict={self.inputs:X_test, 
                                                                                  self.labels:y_test})
                        print("test loss: {:.4f}\t|  test accuracy: {:.2f}%".format(test_pred_total_loss, 
                                                                                    test_pred_acc*100))
                        if test_pred_acc >= best_acc:
                            best_acc = test_pred_acc
                            self.save("./my_model/"+self._num_+"/train_model.ckpt")
            return self
    
    def save(self, path):
        self._saver.save(self._session, path)
    
    def restore(self):
        path = "./my_model/"+self._num_+"/train_model.ckpt"
        self._saver.restore(self._session, path)

    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self.predictor.eval(feed_dict={self.inputs: X})
    
    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([self.classes_[class_index] for class_index in class_indices], np.int32).reshape(-1)
