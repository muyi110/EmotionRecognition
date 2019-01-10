# -*- coding:UTF-8 -*-
################################################################
# 此部分实现 MCD_DA 网络的构建，特征提取基于 TCN 网络
################################################################
import tensorflow as tf
import numpy as np
from tcn import TCN
from sklearn.exceptions import NotFittedError
from gradient_reversal import gradient_reversal
from utils import batch_generator

class MCD_DA_Model():
    def __init__(self, 
                 sequence_length, 
                 kernel_size, 
                 num_channels, 
                 dropout,
                 batch_size,
                 in_channels,
                 num_k = 3,
                 random_state=None, 
                 reverse=False,
                 optimizer_class=tf.train.AdamOptimizer):
        self.sequence_length = sequence_length
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.dropout = dropout
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.random_state = random_state
        self.reverse=reverse # 用于判断是否使用梯度翻转层 (GRL)
        self.optimizer_class = optimizer_class
        self.num_k = num_k
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
        self._training = tf.placeholder_with_default(False, shape=(), name="training")
        self.flag = tf.placeholder_with_default(False, shape=(), name="flag")
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        self.l = tf.placeholder(tf.float32, shape=(), name="l") # 用于梯度翻转层常数
        # TCN 模型用于提取特征
        with tf.variable_scope("feature_extractor"):
            # feature.shape = (samples, seq_length, features)
            features = TCN(self.inputs, n_outputs, self.num_channels, self.sequence_length, 
                           self.kernel_size, self.dropout, is_training=self._training)
            # 取最后一个输出作为最终特征
            self.feature = features[:, -1, :]
            # self.feature = tf.contrib.layers.flatten(features)
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
        with tf.variable_scope("label_predictor_one"):
            if self.reverse:  # 是否使用梯度翻转层训练网络
                self.feature = gradient_reversal(self.feature, self.l)
            temp_one = tf.contrib.layers.fully_connected(self.feature, 30, activation_fn=tf.nn.relu)
            temp_one = tf.contrib.layers.dropout(temp_one, 1-self.dropout, is_training=self._training)
            _all_logits = tf.contrib.layers.fully_connected(temp_one, n_outputs, activation_fn=None)
            all_logits = lambda: _all_logits
            s_logits = lambda: tf.slice(_all_logits, [0, 0], [self.batch_size//2, -1])
            t_logits = lambda: tf.slice(_all_logits, [self.batch_size//2, 0], [self.batch_size//2, -1])
            logits = tf.cond(self.flag, s_logits, all_logits)

            all_labels = lambda: self.labels
            source_labels = lambda: tf.slice(self.labels, [0], [self.batch_size//2])
            self.classify_labels = tf.cond(self.flag, source_labels, all_labels)

            self.predictor_one = tf.nn.softmax(logits)
            self.predictor_one_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.classify_labels, 
                                                                                     logits=logits)
            self.t_logits_one = tf.cond(self.flag, t_logits, all_logits)
        # softmax 用于分类预测
        with tf.variable_scope("label_predictor_two"):
            if self.reverse:  # 是否使用梯度翻转层训练网络
                self.feature = gradient_reversal(self.feature, self.l)
            temp_two = tf.contrib.layers.fully_connected(self.feature, 30, activation_fn=tf.nn.relu, 
                                                         weights_initializer=tf.random_normal_initializer(0, 0.01))
            temp_two = tf.contrib.layers.dropout(temp_two, 1-self.dropout, is_training=self._training)
            _all_logits_2 = tf.contrib.layers.fully_connected(temp_two, n_outputs, activation_fn=None,
                                                              weights_initializer=tf.random_normal_initializer(0, 0.01))
            all_logits_2 = lambda: _all_logits_2
            s_logits_2 = lambda: tf.slice(_all_logits_2, [0, 0], [self.batch_size//2, -1])
            t_logits_2 = lambda: tf.slice(_all_logits_2, [self.batch_size//2, 0], [self.batch_size//2, -1])
            logits_2 = tf.cond(self.flag, s_logits_2, all_logits_2)

            all_labels_2 = lambda: self.labels
            source_labels_2 = lambda: tf.slice(self.labels, [0], [self.batch_size//2])
            self.classify_labels_2 = tf.cond(self.flag, source_labels_2, all_labels_2)

            self.predictor_two = tf.nn.softmax(logits_2)
            self.predictor_two_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.classify_labels_2, 
                                                                                     logits=logits_2)
            self.t_logits_two = tf.cond(self.flag, t_logits_2, all_logits_2)
        
    def _set_optimizer_parameters(self):
        # 特征提取器优化的参数列表
        self.opt_g_parameter = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="feature_extractor/")
        # 分类器1 优化的参数列表
        self.opt_c1_parameter = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="label_predictor_one/")
        # 分类器2 优化的参数列表
        self.opt_c2_parameter = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="label_predictor_two/")
        # 分类器1 和分类器2 的参数列表
        self.opt_c_parameter = self.opt_c1_parameter + self.opt_c2_parameter
        # 所有的变量
        self.opt_all_parameter = self.opt_g_parameter + self.opt_c_parameter

    def _discrepancy(self, out1, out2):
        t_out1 = tf.nn.softmax(out1)
        t_out2 = tf.nn.softmax(out2)
        return tf.reduce_mean(tf.reduce_mean(tf.abs(t_out1-t_out2), axis=1))

    def _build_graph(self, n_outputs):
        self._build_model(n_outputs)
        self._set_optimizer_parameters()
        # 构建损失节点
        one_loss = tf.reduce_mean(self.predictor_one_loss)
        two_loss = tf.reduce_mean(self.predictor_two_loss)
        source_loss = one_loss + two_loss                                      # 源域的分类损失
        target_loss = self._discrepancy(self.t_logits_one, self.t_logits_two)  # 目标域损失
        total_loss = source_loss - target_loss
        # 构建训练节点
        step_one_train_op = self.optimizer_class(self.learning_rate).minimize(source_loss, var_list=self.opt_all_parameter)
        step_two_train_op = self.optimizer_class(self.learning_rate).minimize(total_loss, var_list=self.opt_c_parameter)
        step_three_train_op = self.optimizer_class(self.learning_rate).minimize(target_loss, var_list=self.opt_g_parameter)
        # 构建评估节点
        correct_label_predictor_one = tf.nn.in_top_k(self.predictor_one, self.classify_labels, 1)
        label_one_acc = tf.reduce_mean(tf.cast(correct_label_predictor_one, tf.float32))
        correct_label_predictor_two = tf.nn.in_top_k(self.predictor_two, self.classify_labels_2, 1)
        label_two_acc = tf.reduce_mean(tf.cast(correct_label_predictor_two, tf.float32))
        # 构建全局初始化节点
        init = tf.global_variables_initializer()
        # 构建模型保存节点
        saver = tf.train.Saver()
    
        self._diff_loss, self._c_one_loss, self._c_two_loss, self._total_loss = target_loss, one_loss, two_loss, total_loss
        self._one_train_op = step_one_train_op
        self._two_train_op = step_two_train_op
        self._three_train_op = step_three_train_op
        self._label_one_acc, self._label_two_acc = label_one_acc, label_two_acc
        self._init, self._saver = init, saver
    
    def close_session(self):
        if self._session:
            self._session.close()

    def fit(self, X, y, epochs, X_test, y_test, outputs, people_num=None):
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
        seed = 0
        best_acc = 0 # 测试集最好的准确率
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            sess.run(self._init)
            #self._saver.restore(sess, "./pre_train_params/pre_train_params.ckpt")
            for epoch in range(epochs):
                lr = 0.0002
                seed += 1
                gen_batch = batch_generator(X, y, X_test, y_test, self.batch_size, seed)
                for batch in gen_batch:
                    X_batch, y_batch = batch
                    # step A
                    sess.run(self._one_train_op, feed_dict={self.inputs:X_batch, self.labels:y_batch, self.flag:True,
                                                            self.learning_rate:lr, self._training:True})
                for batch in gen_batch:
                    X_batch, y_batch = batch
                    # step B
                    sess.run(self._two_train_op, feed_dict={self.inputs:X_batch, self.labels:y_batch, self.flag:True,
                                                            self.learning_rate:lr, self._training:True})
                for batch in gen_batch:
                    X_batch, y_batch = batch
                    # step C
                    for i in range(self.num_k):
                        sess.run(self._three_train_op, feed_dict={self.inputs:X_batch, self.labels:y_batch, self.flag:True,
                                                                  self.learning_rate:lr, self._training:True})
                # 打印消息
                if epoch % 1 == 0:
                    c_one_loss, c_two_loss, one_acc, two_acc = sess.run([self._c_one_loss, 
                                                                         self._c_two_loss, self._label_one_acc, 
                                                                         self._label_two_acc], 
                                                                         feed_dict={self.inputs:X, 
                                                                                    self.labels:y})
                    print("{}  Training accuracy: one: {:.2f}%\ttwo: {:.2f}%\tLoss1: {:.4f}\tLoss2: {:.4f}.".format(epoch, 
                          one_acc*100, two_acc*100, c_one_loss, c_two_loss))
                    diff_loss, total_loss = sess.run([self._diff_loss, self._total_loss], feed_dict={self.inputs:X_batch,
                                                                                                     self.labels:y_batch,
                                                                                                     self.flag:True})
                    print("Total loss: {:.4f}, diff_loss: {:.4f}.".format(total_loss, diff_loss))
                    test_one_acc, test_two_acc, test_one_loss, test_two_loss = sess.run([self._label_one_acc,
                                                                                         self._label_two_acc,
                                                                                         self._c_one_loss,
                                                                                         self._c_two_loss],
                                                                                         feed_dict={self.inputs:X_test,
                                                                                                    self.labels:y_test})
                    print("Test accuracy: one: {:.2f}%\ttwo: {:.2f}%.  |  Test loss: one: {:.4f}\ttwo: {:.4f}.".format(
                          test_one_acc*100, test_two_acc*100, test_one_loss, test_two_loss))
                    #if one_acc >= 0.99 and two_acc >= 0.99:
                    #    self.save("./pre_train_params/pre_train_params.ckpt")
                    if test_one_acc > best_acc:
                        best_acc = test_one_acc
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
