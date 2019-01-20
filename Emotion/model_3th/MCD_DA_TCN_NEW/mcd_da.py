# -*- coding:UTF-8 -*-
################################################################
# 此部分实现 MCD_DA 网络的构建，特征提取基于 TCN 网络
################################################################
import tensorflow as tf
import numpy as np
from tcn import TCN
from sklearn.exceptions import NotFittedError
from utils import batch_generator

class MCD_DA_Model():
    def __init__(self, 
                 sequence_length, 
                 kernel_size, 
                 num_channels, 
                 dropout,
                 batch_size,
                 in_channels,
                 num_k = 20,
                 random_state=None, 
                 optimizer_class=tf.train.AdamOptimizer):
        self.sequence_length = sequence_length
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.dropout = dropout
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.random_state = random_state
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
        self.inputs_s = tf.placeholder(tf.float32, shape=(None, self.sequence_length, self.in_channels), name="inputs_s")
        self.labels_s = tf.placeholder(tf.int32, shape=(None), name="labels_s")
        self.inputs_t = tf.placeholder(tf.float32, shape=(None, self.sequence_length, self.in_channels), name="inputs_t")
        self.labels_t = tf.placeholder(tf.int32, shape=(None), name="labels_t")
        self._training = tf.placeholder_with_default(False, shape=(), name="training")
        self.flag = tf.placeholder_with_default(False, shape=(), name="flag")
        self.flag_test = tf.placeholder_with_default(False, shape=(), name="flag_test")
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        # TCN 模型用于提取特征
        with tf.variable_scope("feature_extractor_s"):
            # feature.shape = (samples, seq_length, features)
            features_s = TCN(self.inputs_s, n_outputs, self.num_channels, self.sequence_length, 
                             self.kernel_size, self.dropout, is_training=self._training)
            # 取最后一个输出作为最终特征
            self.feature_s = features_s[:, -1, :]

        with tf.variable_scope("feature_extractor_t"):
            # feature.shape = (samples, seq_length, features)
            features_t = TCN(self.inputs_t, n_outputs, self.num_channels, self.sequence_length, 
                             self.kernel_size, self.dropout, is_training=self._training)
            # 取最后一个输出作为最终特征
            self.feature_t = features_t[:, -1, :]

        # 将源域特征和目标域特征进行级联
        self.feature = tf.concat([self.feature_s, self.feature_t], axis=0)
        # softmax 用于分类预测
        with tf.variable_scope("label_predictor_one"):
            temp_one = tf.contrib.layers.fully_connected(self.feature, 30, activation_fn=tf.nn.relu)
            temp_one = tf.contrib.layers.dropout(temp_one, 1-self.dropout, is_training=self._training)
            logits = tf.contrib.layers.fully_connected(temp_one, n_outputs, activation_fn=None)

            classify_logits = tf.cond(self.flag, lambda:logits[:self.batch_size], lambda:logits[:1664])
            self.classify_labels = tf.cond(self.flag_test, lambda:self.labels_t, lambda:self.labels_s)

            self.predictor_one_s = tf.nn.softmax(logits[:1664]) # 用于训练集评估准确率
            self.predictor_one_t = tf.nn.softmax(logits[-416:]) # 用于测试集评估准确率
            self.predictor_one_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.classify_labels, 
                                                                                     logits=classify_logits)
            self.one_logits_t = logits[self.batch_size:]
            self.one = tf.nn.softmax(self.one_logits_t)

        with tf.variable_scope("label_predictor_two"):
            temp_two = tf.contrib.layers.fully_connected(self.feature, 30, activation_fn=tf.nn.relu, 
                                                         weights_initializer=tf.random_normal_initializer(0, 0.01))
            temp_two = tf.contrib.layers.dropout(temp_two, 1-self.dropout, is_training=self._training)
            logits_2 = tf.contrib.layers.fully_connected(temp_two, n_outputs, activation_fn=None,
                                                         weights_initializer=tf.random_normal_initializer(0, 0.01))
            classify_logits_2 = tf.cond(self.flag, lambda:logits_2[:self.batch_size], lambda:logits_2[:1664])

            self.predictor_two_s = tf.nn.softmax(logits_2[:1664]) # 用于训练集评估准确率
            self.predictor_two_t = tf.nn.softmax(logits_2[-416:]) # 用于测试集评估准确率
            self.predictor_two_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.classify_labels, 
                                                                                     logits=classify_logits_2)
            self.two_logits_t = logits_2[self.batch_size:]
            self.two = tf.nn.softmax(self.two_logits_t)

    def _set_optimizer_parameters(self):
        # 特征提取器优化的参数列表
        self.opt_g_s_parameter = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="feature_extractor_s/")
        self.opt_g_t_parameter = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="feature_extractor_t/")
        # 分类器1 优化的参数列表
        self.opt_c1_parameter = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="label_predictor_one/")
        # 分类器2 优化的参数列表
        self.opt_c2_parameter = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="label_predictor_two/")
        # 分类器1 和分类器2 的参数列表
        self.opt_c_parameter = self.opt_c1_parameter + self.opt_c2_parameter
        # 所有的变量
        self.opt_s_all_parameter = self.opt_g_s_parameter + self.opt_c_parameter

    def _discrepancy(self, out1, out2):
        t_out1 = tf.nn.softmax(out1)
        t_out2 = tf.nn.softmax(out2)
        return tf.reduce_mean(tf.reduce_mean(tf.abs(t_out1-t_out2), axis=1))

    def _coral_loss(self, out_s, out_t):
        out_s = out_s - tf.reduce_mean(out_s, axis=0)
        out_t = out_t - tf.reduce_mean(out_t, axis=0)
        cov_s = (1./(self.batch_size-1))*tf.matmul(out_s, out_s, transpose_a=True)
        cov_t = (1./(self.batch_size-1))*tf.matmul(out_t, out_t, transpose_a=True)
        return tf.reduce_mean(tf.abs(tf.subtract(cov_s, cov_t)))

    def _build_graph(self, n_outputs):
        self._build_model(n_outputs)
        self._set_optimizer_parameters()
        # 构建损失节点
        #diff=tf.reduce_mean(self.feature_s, 0) - tf.reduce_mean(self.feature_t, 0)
        #self.domain_loss=tf.reduce_sum(tf.multiply(diff,diff))
        self.domain_loss = self._coral_loss(self.feature_s, self.feature_t)

        one_loss = tf.reduce_mean(self.predictor_one_loss)
        two_loss = tf.reduce_mean(self.predictor_two_loss)
        source_loss = one_loss + two_loss                                      # 源域的分类损失
        target_loss = self._discrepancy(self.one_logits_t, self.two_logits_t)  # 目标域损失
        total_loss = source_loss - target_loss
        # 构建训练节点
        step_one_train_op = self.optimizer_class(self.learning_rate).minimize(source_loss, 
                                                                              var_list=self.opt_s_all_parameter)
        step_two_train_op = self.optimizer_class(self.learning_rate).minimize(source_loss, var_list=self.opt_c_parameter)
        step_three_train_op = self.optimizer_class(self.learning_rate).minimize(self.domain_loss, 
                                                                                var_list=self.opt_g_t_parameter)
        # 构建评估节点
        correct_label_predictor_one_s = tf.nn.in_top_k(self.predictor_one_s, self.labels_s, 1)
        label_one_s_acc = tf.reduce_mean(tf.cast(correct_label_predictor_one_s, tf.float32))
        correct_label_predictor_two_s = tf.nn.in_top_k(self.predictor_two_s, self.labels_s, 1)
        label_two_s_acc = tf.reduce_mean(tf.cast(correct_label_predictor_two_s, tf.float32))
        
        correct_label_predictor_one_t = tf.nn.in_top_k(self.predictor_one_t, self.labels_t, 1)
        label_one_t_acc = tf.reduce_mean(tf.cast(correct_label_predictor_one_t, tf.float32))
        correct_label_predictor_two_t = tf.nn.in_top_k(self.predictor_two_t, self.labels_t, 1)
        label_two_t_acc = tf.reduce_mean(tf.cast(correct_label_predictor_two_t, tf.float32))
        # 构建全局初始化节点
        init = tf.global_variables_initializer()
        # 构建模型保存节点
        saver = tf.train.Saver()
    
        self._diff_loss, self._c_one_loss, self._c_two_loss, self._total_loss = target_loss, one_loss, two_loss, total_loss
        self._one_train_op = step_one_train_op
        self._two_train_op = step_two_train_op
        self._three_train_op = step_three_train_op
        self._label_one_s_acc, self._label_two_s_acc = label_one_s_acc, label_two_s_acc
        self._label_one_t_acc, self._label_two_t_acc = label_one_t_acc, label_two_t_acc
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
            for epoch in range(epochs):
                lr = 0.0002
                seed += 1
                gen_batch = batch_generator(X, y, X_test, y_test, self.batch_size, seed)
                if epoch < 200:
                    for batch in gen_batch:
                        X_batch_s, y_batch_s, X_batch_t, y_batch_t = batch
                        # step A
                        sess.run(self._one_train_op, feed_dict={self.inputs_s:X_batch_s, self.labels_s:y_batch_s,
                                                                self.inputs_t:X_batch_t, self.labels_t:y_batch_t,
                                                                self.flag:True, self.learning_rate:lr, self._training:True})
                if epoch >= 200 and epoch < 300:
                    for batch in gen_batch:
                        X_batch_s, y_batch_s, X_batch_t, y_batch_t = batch
                        # step B
                        sess.run(self._two_train_op, feed_dict={self.inputs_s:X_batch_s, self.labels_s:y_batch_s,
                                                                self.inputs_t:X_batch_t, self.labels_t:y_batch_t,
                                                                self.flag:True, self.learning_rate:lr, self._training:True})
                if epoch >= 300:
                    for batch in gen_batch:
                        X_batch_s, y_batch_s, X_batch_t, y_batch_t = batch
                        # step C
                        for i in range(self.num_k):
                            sess.run(self._three_train_op, feed_dict={self.inputs_s:X_batch_s, self.labels_s:y_batch_s,
                                                                      self.inputs_t:X_batch_t, self.labels_t:y_batch_t,
                                                                      self.learning_rate:lr, self._training:True, 
                                                                      self.flag:True})
                # 打印消息
                if epoch % 1 == 0:
                    c_one_loss, c_two_loss, one_acc, two_acc = sess.run([self._c_one_loss, 
                                                                         self._c_two_loss, self._label_one_s_acc, 
                                                                         self._label_two_s_acc], 
                                                                         feed_dict={self.inputs_s:X, self.inputs_t:X_test,
                                                                                    self.labels_s:y, self.labels_t:y_test})
                    print("{}  Training accuracy: one: {:.2f}%\ttwo: {:.2f}%\tLoss1: {:.4f}\tLoss2: {:.4f}.".format(epoch, 
                          one_acc*100, two_acc*100, c_one_loss, c_two_loss))
                    diff_loss, domain_loss, two, one = sess.run([self._diff_loss, self.domain_loss, self.two, self.one], feed_dict={self.inputs_s:X_batch_s, self.inputs_t:X_batch_t})
                    print("diff loss: {:.4f}   domain loss {:.4f}".format(diff_loss, domain_loss))
                    print(two[:10], one[:10])
                    test_one_acc, test_two_acc = sess.run([self._label_one_t_acc,self._label_two_t_acc],
                                                           feed_dict={self.inputs_t:X_test, self.labels_t:y_test,
                                                                      self.inputs_s:X, self.labels_s:y})
                    print("Test accuracy: one: {:.2f}%\ttwo: {:.2f}%".format(test_one_acc*100, test_two_acc*100))
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
