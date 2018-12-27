# -*- coding:UTF-8 -*-
################################################################
# 此部分实现 DANN 网络的构建，特征提取基于 TCN 网络
################################################################
import tensorflow as tf
import numpy as np
from tcn import TCN
from sklearn.exceptions import NotFittedError
from gradient_reversal import gradient_reversal
#from tensorflow.contrib.layers import batch_norm
from utils import batch_generator

class DANNModel():
    def __init__(self, 
                 sequence_length, 
                 kernel_size, 
                 num_channels, 
                 dropout,
                 batch_size,
                 in_channels,
                 train_ratio=0.5,
                 random_state=None, 
                 optimizer_class1=tf.train.RMSPropOptimizer,
                 optimizer_class2=tf.train.GradientDescentOptimizer):
                 #optimizer_class1=tf.train.AdamOptimizer):
        self.sequence_length = sequence_length
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.dropout = dropout
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.train_ratio = train_ratio
        self.random_state = random_state
        self.optimizer_class1 = optimizer_class1
        self.optimizer_class2 = optimizer_class2
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
        self.domain_labels = tf.placeholder(tf.float32, shape=(None, 1), name="domain_labels")
        self._training = tf.placeholder_with_default(False, shape=(), name="training")
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        self.l = tf.placeholder(tf.float32, shape=(), name="l") # 用于梯度翻转层常数
        # bn_params = {"is_training":self._training, 'decay':0.99, 'updates_collections':None}
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
            all_features = lambda: self.feature
            source_features = lambda: tf.slice(self.feature, [0, 0], [int(self.batch_size*self.train_ratio), -1])
            classify_features = tf.cond(self._training, source_features, all_features)

            all_labels = lambda: self.labels
            source_labels = lambda: tf.slice(self.labels, [0], [int(self.batch_size*self.train_ratio)])
            self.classify_labels = tf.cond(self._training, source_labels, all_labels)

            # p_temp_0 = tf.contrib.layers.fully_connected(classify_features, 20, activation_fn=tf.nn.relu)
            logits = tf.contrib.layers.fully_connected(classify_features, n_outputs, activation_fn=None) 
            self.predictor = tf.nn.softmax(logits, name="predictor")
            self.predictor_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.classify_labels, 
                                                                                 logits=logits)
        # softmax 用于域判别器
        with tf.variable_scope("domain_predictor"):
            # 反向传播时候翻转梯度
            feat = gradient_reversal(self.feature, self.l)
            d_temp = tf.contrib.layers.fully_connected(feat, 30, activation_fn=tf.nn.relu, scope="d_temp")
            # d_temp_0 = tf.contrib.layers.dropout(d_temp, 1-0.5, is_training=self._training)
            d_logits = tf.contrib.layers.fully_connected(d_temp, 1, activation_fn=None, scope="d_logits")
            self.domain_predictor = tf.sigmoid(d_logits, name="domain_pre")
            # self.domain_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=d_logits, labels=self.domain_labels)
            self.domain_loss = tf.losses.log_loss(predictions=self.domain_predictor, labels=self.domain_labels)
            # self.domain_loss = tf.losses.hinge_loss(logits=d_logits, labels=self.domain_labels)
            
            # 下面开始计算源域样本在判别器上的损失
            # self.domain_predictor_source = self.domain_predictor[:int(self.batch_size*self.train_ratio)]
            self.domain_predictor_source = self.domain_predictor[:]
            self.domain_loss_source = tf.losses.mean_squared_error(predictions=self.domain_predictor_source, labels=tf.subtract(self.domain_labels[:], 0.5))
            #self.domain_loss_source = tf.losses.mean_squared_error(predictions=self.domain_predictor_source, labels=tf.subtract(self.domain_labels[:int(self.batch_size*self.train_ratio)], 0.5))

    def _build_graph(self, n_outputs):
        self._build_model(n_outputs)
        # 构建损失节点
        pred_loss = tf.reduce_mean(self.predictor_loss)
        domain_loss_ = tf.reduce_mean(self.domain_loss)
        domain_loss_source = tf.multiply(tf.reduce_mean(self.domain_loss_source), 1)
        total_loss = tf.add(pred_loss, domain_loss_)
        # 构建训练节点
        pre_train_op = self.optimizer_class2(learning_rate=self.learning_rate).minimize(pred_loss)
        dann_train_op = self.optimizer_class1(learning_rate=self.learning_rate).minimize(total_loss)
        domain_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                         scope="domain_predictor/d_temp|domain_predictor/d_logits|domain_predictor/d_temp1")
        domain_train_op = self.optimizer_class1(learning_rate=self.learning_rate).minimize(domain_loss_) 
                                                                                           #var_list=domain_train_vars)
        # 构建评估节点
        correct_label_predictor = tf.nn.in_top_k(self.predictor, self.classify_labels, 1)
        label_acc = tf.reduce_mean(tf.cast(correct_label_predictor, tf.float32))
        correct_domain_predictor = tf.equal(self.domain_labels, tf.cast(tf.greater(self.domain_predictor, 0.5), tf.float32))
        domain_acc = tf.reduce_mean(tf.cast(correct_domain_predictor, tf.float32))
        # 构建全局初始化节点
        init = tf.global_variables_initializer()
        # 构建模型保存节点
        saver = tf.train.Saver()

        self._pred_loss, self._domain_loss, self._total_loss = pred_loss, domain_loss_, total_loss
        self._regular_train_op, self._dann_train_op, self._domain_train_op = pre_train_op, dann_train_op, domain_train_op
        self._label_acc, self._domain_acc = label_acc, domain_acc
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
        epoch_count = 0
        k = 3
        with self._session.as_default() as sess:
            sess.run(self._init)
            # batch 生成器
            gen_source_batch = batch_generator([X, y], int(self.batch_size*self.train_ratio))
            gen_target_batch = batch_generator([X_test, y_test], self.batch_size-int(self.batch_size*self.train_ratio))
            gen_source_only_batch = batch_generator([X, y], self.batch_size)
            # domain_label = np.vstack([np.tile([1, 0], [int(self.batch_size*self.train_ratio), 1]), 
            #                           np.tile([0, 1], [self.batch_size-int(self.batch_size*self.train_ratio), 1])])
            temp_list=[1]*int(self.batch_size*self.train_ratio)+[0]*(self.batch_size-int(self.batch_size*self.train_ratio))
            domain_label = np.array(temp_list).reshape(-1, 1)
            for step in range(num_steps):
                p = float(step) / num_steps
                l = 2. / (1. + np.exp(-10*p)) - 1
                lr = 0.008 / (1 + 10 * p)**0.75
                
                if training_mode == "dann":
                    X0, y0 = next(gen_source_batch)
                    X1, y1 = next(gen_target_batch)
                    X_batch = np.vstack([X0, X1])
                    y_batch = np.hstack([y0, y1])
                    assert(y_batch.shape==(self.batch_size,))
                    assert(X_batch.shape==(self.batch_size, self.sequence_length, self.in_channels))
                    if k > 0:
                        k -= 1
                        sess.run(self._dann_train_op, feed_dict={self.inputs:X_batch, self.labels:y_batch, 
                                                                  self.domain_labels:domain_label, self._training:True, 
                                                                  self.l:l, self.learning_rate:lr})
                    else:
                        k = 3
                        sess.run(self._dann_train_op, feed_dict={self.inputs:X_batch, self.labels:y_batch, 
                                                                    self.domain_labels:domain_label, self._training:True, 
                                                                    self.l:l, self.learning_rate:lr})
                    # print info
                    if step % (len(y)//(self.batch_size*self.train_ratio)) == 0 and step != 0:
                        epoch_count += 1
                        train_pred_total_loss, train_pred_acc = sess.run([self._pred_loss, self._label_acc], 
                                                                          feed_dict={self.inputs:X, self.labels:y})
                        domain_loss, domain_acc, p = sess.run([self._domain_loss, self._domain_acc, 
                                                               self.domain_predictor], 
                                                               feed_dict={self.inputs:X_batch, self.labels:y_batch, 
                                                               self.domain_labels:domain_label})
                        print("{}\ttraining loss: {:.4f}\t|  trainning accuracy: {:.2f}%".format(epoch_count, 
                              train_pred_total_loss, train_pred_acc*100))
                        print(p[-10:])
                        print("domain loss: {:.2f}\t|  domain accuracy: {:.2f}%".format(domain_loss, domain_acc*100))
                        test_pred_total_loss, test_pred_acc = sess.run([self._pred_loss, self._label_acc], 
                                                                       feed_dict={self.inputs:X_test, 
                                                                                  self.labels:y_test})
                        if test_pred_acc >= best_acc:
                            best_acc = test_pred_acc
                            self.save("./my_model/"+self._num_+"/train_model.ckpt")
                        print("test loss: {:.4f}\t|  test accuracy: {:.2f}%".format(test_pred_total_loss, 
                                                                                    test_pred_acc*100))
                if training_mode == "source":
                    X_batch, y_batch = next(gen_source_only_batch)
                    sess.run(self._regular_train_op, feed_dict={self.inputs:X_batch, self.labels:y_batch, 
                                                                self.learning_rate:lr})
                    # print info
                    if step % (len(y)//(self.batch_size)) == 0 and step != 0:
                        epoch_count += 1
                        train_pred_total_loss, train_pred_acc = sess.run([self._pred_loss, self._label_acc], 
                                                                          feed_dict={self.inputs:X, self.labels:y})
                        print("{}\ttraining loss: {:.4f}\t|  trainning accuracy: {:.2f}%".format(epoch_count, 
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
