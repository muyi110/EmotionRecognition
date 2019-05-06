# -*- coding:UTF-8 -*-
import tensorflow as tf
import numpy as np
from tcn import TCN
from utils import batch_generator
from gradient_reversal import gradient_reversal
from sklearn.exceptions import NotFittedError

class MMJLNNModel(): # 多模态联合学习神经网络
    def __init__(self, 
                 sequence_length,
                 kernel_size,
                 num_channels,
                 dropout,
                 batch_size,
                 in_channels,
                 number_of_class_one,
                 n_steps = 9, # 用于RNN网络
                 n_neurons = 32, # 用于RNN网络
                 random_state=None,
                 optimizer_class=tf.train.AdamOptimizer):
        self.sequence_length = sequence_length
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.dropout = dropout
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.number_of_class_one = number_of_class_one
        self.random_state = random_state
        self.optimizer_class = optimizer_class
        self.n_steps = n_steps
        self.n_neurons = n_neurons
        self._session = None

    #def residual_block(self, inputs, out_channels, filter_size):
    #    conv1d_out = tf.layers.conv1d(inputs=inputs, filters=out_channels, kernel_size=filter_size)
        

    def _build_model(self, n_outputs):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)
        # PER
        self.EOG_inputs = tf.placeholder(tf.float32, shape=(None, self.n_steps, 256), name="EOG_inputs")
        self.EMG_inputs = tf.placeholder(tf.float32, shape=(None, self.n_steps, 256), name="EMG_inputs")
        self.GSR_inputs = tf.placeholder(tf.float32, shape=(None, self.n_steps, 128), name="GSR_inputs")
        self.RSP_inputs = tf.placeholder(tf.float32, shape=(None, self.n_steps, 128), name="RSP_inputs")
        self.BLV_inputs = tf.placeholder(tf.float32, shape=(None, self.n_steps, 128), name="BLV_inputs")
        self.TMR_inputs = tf.placeholder(tf.float32, shape=(None, self.n_steps, 128), name="TMR_inputs")
        # EEG
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.sequence_length, self.in_channels), 
                                     name="inputs")
        self.labels = tf.placeholder(tf.int32, shape=(None), name='labels')
        # 类别1情绪对应的域标签（每一个实验代表一个域）
        self.domain_labels_class_one = tf.placeholder(tf.int32, shape=(None), name="domain_labels_class_one")
        # 类别2情绪对应的域标签（每一个实验代表一个域）
        self.domain_labels_class_two = tf.placeholder(tf.int32, shape=(None), name="domain_labels_class_two")
        self._training = tf.placeholder_with_default(False, shape=(), name="training")
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        self.l = tf.placeholder(tf.float32, shape=(), name="l") # 用于梯度翻转层常数
        self.number = self.number_of_class_one * (self.batch_size//32)
        # TCN 模型用于提取特征
        with tf.variable_scope("EEG_feature_extractor"):
            # features 的 shape=(samples, sequence_length, features)
            features = TCN(self.inputs, n_outputs, self.num_channels, self.sequence_length, 
                           self.kernel_size, self.dropout, is_training=self._training)
            self.feature = features[:, -1, :]
            self.feature_norm = tf.layers.batch_normalization(self.feature, training=self._training) # BN层
        with tf.variable_scope("EEG_confidence"):
            EEG_output_temp = tf.contrib.layers.fully_connected(self.feature, 32, activation_fn=tf.nn.relu)
            EEG_logits = tf.contrib.layers.fully_connected(EEG_output_temp, 2, activation_fn=None)
            self.EEG_confidence = tf.nn.softmax(EEG_logits, name="EEG_confidence")[:,0:1] # softmax的第一个输出为置信得分
        # 类别1情绪对应的域分类器
        with tf.variable_scope("domain_predictor_class_one"):
            # 一个 batch 中情绪类别1对应的特征
            features_class_one = self.feature[:self.number]
            # 反向传播时翻转梯度
            feat_domain_class_one = gradient_reversal(features_class_one, self.l)
            domain_class_one_temp = tf.contrib.layers.fully_connected(feat_domain_class_one, 32, activation_fn=tf.nn.relu)
            logits_domain_class_one = tf.contrib.layers.fully_connected(domain_class_one_temp, np.asscalar(self.number//4), 
                                                                        activation_fn=None)
            self.predictor_domain_class_one = tf.nn.softmax(logits_domain_class_one, name="predictor_domain_class_one")
            self.domain_loss_class_one = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_domain_class_one,
                                                                                        labels=self.domain_labels_class_one)
        # 类别2情绪对应的域分类器
        with tf.variable_scope("domain_predictor_class_two"):
            # 一个 batch 中情绪类别2对应的特征
            features_class_two = self.feature[self.number:]
            # 反向传播时翻转梯度
            feat_domain_class_two = gradient_reversal(features_class_two, self.l)
            domain_class_two_temp = tf.contrib.layers.fully_connected(feat_domain_class_two, 32, activation_fn=tf.nn.relu)
            logits_domain_class_two = tf.contrib.layers.fully_connected(domain_class_two_temp, np.asscalar(32-self.number//4), 
                                                                        activation_fn=None)
            self.predictor_domain_class_two = tf.nn.softmax(logits_domain_class_two, name="predictor_domain_class_two")
            self.domain_loss_class_two = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_domain_class_two,
                                                                                        labels=self.domain_labels_class_two)
        # EOG
        with tf.variable_scope("EOG_feature_extractor"):
            #EOG_conv_output_1 = tf.layers.conv1d(inputs=self.EOG_inputs, filters=100, kernel_size=1, activation=tf.nn.relu)
            #EOG_conv_output_1_drop = tf.contrib.layers.dropout(EOG_conv_output_1, 0.8, is_training=self._training) 
            #EOG_conv_output = tf.layers.conv1d(inputs=EOG_conv_output_1_drop, filters=20, kernel_size=1, activation=tf.nn.relu)

            #EOG_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.n_neurons, activation=tf.nn.relu)
            #EOG_outputs, EOG_states = tf.nn.dynamic_rnn(EOG_lstm_cell, EOG_conv_output, dtype=tf.float32)
            #self.EOG_feature = EOG_states[1]
            EOG_features = TCN(self.EOG_inputs, 2, [512, 256, 128], 9, 
                           self.kernel_size, 0.8, is_training=self._training)
            self.EOG_feature = EOG_features[:, -1, :]
            self.EOG_feature_norm = tf.layers.batch_normalization(self.EOG_feature, training = self._training) # BN层
        with tf.variable_scope("EOG_confidence"):
            EOG_output_temp = tf.contrib.layers.fully_connected(self.EOG_feature, 32, activation_fn=tf.nn.relu)
            EOG_logits = tf.contrib.layers.fully_connected(EOG_output_temp, 2, activation_fn=None)
            self.EOG_confidence = tf.nn.softmax(EOG_logits, name="EOG_confidence")[:,0:1] # softmax的第一个输出为置信得分
        # EMG
        with tf.variable_scope("EMG_feature_extractor"):
            #EMG_conv_output_1 = tf.layers.conv1d(inputs=self.EMG_inputs, filters=100, kernel_size=1, activation=tf.nn.relu)
            #EMG_conv_output_1_drop = tf.contrib.layers.dropout(EMG_conv_output_1, 0.8, is_training=self._training) 
            #EMG_conv_output = tf.layers.conv1d(inputs=EMG_conv_output_1_drop, filters=20, kernel_size=1, activation=tf.nn.relu)

            #EMG_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.n_neurons, activation=tf.nn.relu)
            #EMG_outputs, EMG_states = tf.nn.dynamic_rnn(EMG_lstm_cell, EMG_conv_output, dtype=tf.float32)
            #self.EMG_feature = EMG_states[1]
            EMG_features = TCN(self.EMG_inputs, 2, [512, 256, 128], 9, 
                           self.kernel_size, 0.8, is_training=self._training)
            self.EMG_feature = EMG_features[:, -1, :]
            self.EMG_feature_norm = tf.layers.batch_normalization(self.EMG_feature, training=self._training) # BN层
        with tf.variable_scope("EMG_confidence"):
            EMG_output_temp = tf.contrib.layers.fully_connected(self.EMG_feature, 32, activation_fn=tf.nn.relu)
            EMG_logits = tf.contrib.layers.fully_connected(EMG_output_temp, 2, activation_fn=None)
            self.EMG_confidence = tf.nn.softmax(EMG_logits, name="EMG_confidence")[:,0:1] # softmax的第一个输出为置信得分
        # GSR
        with tf.variable_scope("GSR_feature_extractor"):
            #GSR_conv_output_1 = tf.layers.conv1d(inputs=self.GSR_inputs, filters=100, kernel_size=1, activation=tf.nn.relu)
            #GSR_conv_output_1_drop = tf.contrib.layers.dropout(GSR_conv_output_1, 0.8, is_training=self._training) 
            #GSR_conv_output = tf.layers.conv1d(inputs=GSR_conv_output_1_drop, filters=20, kernel_size=1, activation=tf.nn.relu)

            #GSR_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.n_neurons, activation=tf.nn.relu)
            #GSR_outputs, GSR_states = tf.nn.dynamic_rnn(GSR_lstm_cell, GSR_conv_output, dtype=tf.float32)
            #self.GSR_feature = GSR_states[1]
            GSR_features = TCN(self.GSR_inputs, 2, [256, 128, 64], 9, 
                           self.kernel_size, 0.8, is_training=self._training)
            self.GSR_feature = GSR_features[:, -1, :]
            self.GSR_feature_norm = tf.layers.batch_normalization(self.GSR_feature, training=self._training) # BN层
        with tf.variable_scope("GSR_confidence"):
            GSR_output_temp = tf.contrib.layers.fully_connected(self.GSR_feature, 32, activation_fn=tf.nn.relu)
            GSR_logits = tf.contrib.layers.fully_connected(GSR_output_temp, 2, activation_fn=None)
            self.GSR_confidence = tf.nn.softmax(GSR_logits, name="GSR_confidence")[:,0:1] # softmax的第一个输出为置信得分
        # RSP
        with tf.variable_scope("RSP_feature_extractor"):
            #RSP_conv_output_1 = tf.layers.conv1d(inputs=self.RSP_inputs, filters=100, kernel_size=1, activation=tf.nn.relu)
            #RSP_conv_output_1_drop = tf.contrib.layers.dropout(RSP_conv_output_1, 0.8, is_training=self._training) 
            #RSP_conv_output = tf.layers.conv1d(inputs=RSP_conv_output_1_drop, filters=20, kernel_size=1, activation=tf.nn.relu)

            #RSP_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.n_neurons, activation=tf.nn.relu)
            #RSP_outputs, RSP_states = tf.nn.dynamic_rnn(RSP_lstm_cell, RSP_conv_output, dtype=tf.float32)
            #self.RSP_feature = RSP_states[1]
            RSP_features = TCN(self.RSP_inputs, 2, [256, 128, 64], 9, 
                           self.kernel_size, 0.8, is_training=self._training)
            self.RSP_feature = RSP_features[:, -1, :]
            self.RSP_feature_norm = tf.layers.batch_normalization(self.RSP_feature, training=self._training) # BN层
        with tf.variable_scope("RSP_confidence"):
            RSP_output_temp = tf.contrib.layers.fully_connected(self.RSP_feature, 32, activation_fn=tf.nn.relu)
            RSP_logits = tf.contrib.layers.fully_connected(RSP_output_temp, 2, activation_fn=None)
            self.RSP_confidence = tf.nn.softmax(RSP_logits, name="RSP_confidence")[:,0:1] # softmax的第一个输出为置信得分
        # BLV
        with tf.variable_scope("BLV_feature_extractor"):
            #BLV_conv_output_1 = tf.layers.conv1d(inputs=self.BLV_inputs, filters=100, kernel_size=1, activation=tf.nn.relu)
            #BLV_conv_output_1_drop = tf.contrib.layers.dropout(BLV_conv_output_1, 0.8, is_training=self._training) 
            #BLV_conv_output = tf.layers.conv1d(inputs=BLV_conv_output_1_drop, filters=20, kernel_size=1, activation=tf.nn.relu)

            #BLV_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.n_neurons, activation=tf.nn.relu)
            #BLV_outputs, BLV_states = tf.nn.dynamic_rnn(BLV_lstm_cell, BLV_conv_output, dtype=tf.float32)
            #self.BLV_feature = BLV_states[1]
            BLV_features = TCN(self.BLV_inputs, 2, [256, 128, 64], 9, 
                           self.kernel_size, 0.8, is_training=self._training)
            self.BLV_feature = BLV_features[:, -1, :]
            self.BLV_feature_norm = tf.layers.batch_normalization(self.BLV_feature, training=self._training) # BN层
        with tf.variable_scope("BLV_confidence"):
            BLV_output_temp = tf.contrib.layers.fully_connected(self.BLV_feature, 32, activation_fn=tf.nn.relu)
            BLV_logits = tf.contrib.layers.fully_connected(BLV_output_temp, 2, activation_fn=None)
            self.BLV_confidence = tf.nn.softmax(BLV_logits, name="BLV_confidence")[:,0:1] # softmax的第一个输出为置信得分
        # TMR
        with tf.variable_scope("TMR_feature_extractor"):
            #TMR_conv_output_1 = tf.layers.conv1d(inputs=self.TMR_inputs, filters=100, kernel_size=1, activation=tf.nn.relu)
            #TMR_conv_output_1_drop = tf.contrib.layers.dropout(TMR_conv_output_1, 0.8, is_training=self._training) 
            #TMR_conv_output = tf.layers.conv1d(inputs=TMR_conv_output_1_drop, filters=20, kernel_size=1, activation=tf.nn.relu)

            #TMR_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.n_neurons, activation=tf.nn.relu)
            #TMR_outputs, TMR_states = tf.nn.dynamic_rnn(TMR_lstm_cell, TMR_conv_output, dtype=tf.float32)
            #self.TMR_feature = TMR_states[1]
            TMR_features = TCN(self.TMR_inputs, 2, [256, 128, 64], 9, 
                           self.kernel_size, 0.8, is_training=self._training)
            self.TMR_feature = TMR_features[:, -1, :]
            self.TMR_feature_norm = tf.layers.batch_normalization(self.TMR_feature, training=self._training) # BN层
        with tf.variable_scope("TMR_confidence"):
            TMR_output_temp = tf.contrib.layers.fully_connected(self.TMR_feature, 32, activation_fn=tf.nn.relu)
            TMR_logits = tf.contrib.layers.fully_connected(TMR_output_temp, 2, activation_fn=None)
            self.TMR_confidence = tf.nn.softmax(TMR_logits, name="TMR_confidence")[:,0:1] # softmax的第一个输出为置信得分
        # 将各个传感器数据开始组合（这里采用级联的方式实现）
        EEG = self.EEG_confidence * self.feature
        EOG = self.EOG_confidence * self.EOG_feature_norm
        EMG = self.EMG_confidence * self.EMG_feature_norm
        GSR = self.GSR_confidence * self.GSR_feature_norm
        RSP = self.RSP_confidence * self.RSP_feature_norm
        BLV = self.BLV_confidence * self.BLV_feature_norm
        TMR = self.TMR_confidence * self.TMR_feature_norm
        self.concat_feature = tf.concat([EEG, EOG, EMG, GSR, RSP, BLV, TMR], axis=1)
        #self.concat_feature = EEG + EOG + EMG + GSR + RSP + BLV + TMR
        #self.concat_feature = self.EOG_feature
        with tf.variable_scope("classifier"):
            concat_logit_1 = tf.contrib.layers.fully_connected(self.concat_feature, 100, activation_fn=tf.nn.relu)
            concat_logit_1_drop = tf.contrib.layers.dropout(concat_logit_1, 0.8, is_training=self._training) 
            #concat_logit_2 = tf.contrib.layers.fully_connected(concat_logit_1, 100, activation_fn=tf.nn.relu)
            concat_logit = tf.contrib.layers.fully_connected(concat_logit_1_drop, 2, activation_fn=None)
            self.predictor = tf.nn.softmax(concat_logit, name="predictor")
            self.predictor_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=concat_logit, labels=self.labels)

    def _build_graph(self, n_outputs):
        self._build_model(n_outputs)
        # 构建损失节点（情绪分类损失， 两个域分类损失， 特征约束损失：同一情绪类别差异小， 不同情绪类别差异大）
        predictor_loss = tf.reduce_mean(self.predictor_loss) # 情绪分类损失
        domain_loss = tf.reduce_mean(self.domain_loss_class_one) + tf.reduce_mean(self.domain_loss_class_two) # 域损失
        feature_loss = self._calculate_feature_loss() # 特征损失
        total_loss = predictor_loss + domain_loss + feature_loss*0.1 # 总的损失节点
        #total_loss = predictor_loss
        # 构建训练节点
        train_op = self.optimizer_class(learning_rate=self.learning_rate).minimize(total_loss)
        # 构建评估节点
        correct_label_predictor = tf.nn.in_top_k(self.predictor, self.labels, 1)
        label_acc = tf.reduce_mean(tf.cast(correct_label_predictor, tf.float32))
        correct_domain_label_class_one_predictor = tf.nn.in_top_k(self.predictor_domain_class_one, 
                                                                  self.domain_labels_class_one, 1)
        domain_class_one_acc = tf.reduce_mean(tf.cast(correct_domain_label_class_one_predictor, tf.float32))
        correct_domain_label_class_two_predictor = tf.nn.in_top_k(self.predictor_domain_class_two, 
                                                                  self.domain_labels_class_two, 1)
        domain_class_two_acc = tf.reduce_mean(tf.cast(correct_domain_label_class_two_predictor, tf.float32))
        # 构建全局初始化节点
        init = tf.global_variables_initializer()
        # 构建模型保存节点
        saver = tf.train.Saver()

        self._predictor_loss, self._total_loss, self._feature_loss = predictor_loss, total_loss, feature_loss
        self._train_op = train_op
        self._label_acc = label_acc
        self._domain_class_one_acc = domain_class_one_acc 
        self._domain_class_two_acc = domain_class_two_acc
        self._init, self._saver = init, saver

    def _calculate_feature_loss(self):
        # 计算不同情绪类别间的 MMD 距离
        diff = tf.reduce_mean(self.feature[:self.number], axis=0) - tf.reduce_mean(self.feature[self.number:], axis=0)
        loss_interclass = tf.maximum(0.0, 1-tf.reduce_sum(tf.square(diff)))
        # 计算同一个情绪类别间的 L1 距离
        feature_class_one = self.feature[:self.number]
        feature_class_two = self.feature[self.number:]
        L1_diff_class_one = 0.0
        for i in range(self.number):
            L1_diff_class_one += tf.reduce_sum(tf.square(feature_class_one[i]-tf.reduce_mean(self.feature[:self.number],0)))
        L1_diff_class_one = L1_diff_class_one / self.number
        L1_diff_class_two = 0.0
        for m in range(32-self.number):
            L1_diff_class_two += tf.reduce_sum(tf.square(feature_class_two[i]-tf.reduce_mean(self.feature[self.number:],0)))
        L1_diff_class_two = L1_diff_class_two / (32-self.number)
        loss_intraclass = L1_diff_class_one + L1_diff_class_two
        return loss_intraclass + loss_interclass

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
        
        EEG_X_test, EOG_X_test, EMG_X_test, GSR_X_test, RSP_X_test, BLV_X_test, TMR_X_test = X_test
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
                lr = 0.0001
                p = float(epoch)/epochs
                l = 2. / (1. + np.exp(-2*p)) - 1 # 用于梯度翻转层常数
                seed += 1
                gen_batch, trail_number_class_one = batch_generator(X, y, self.batch_size, seed)
                assert(trail_number_class_one == self.number//4)
                # 构建情绪类别1和情绪类别2的域标签（每一个实验代表一个域）
                number_sample_each_trial = self.batch_size // 32 # 获取一个 batch 中每一个实验选取的样本个数
                temp_list_class_one = []
                for index in range(trail_number_class_one):
                    temp_list_class_one += [index]*number_sample_each_trial
                domain_labels_class_one = np.array(temp_list_class_one)
                temp_list_class_two = []
                for ind in range(32 - trail_number_class_one):
                    temp_list_class_two += [ind]*number_sample_each_trial
                domain_labels_class_two = np.array(temp_list_class_two)
                for batch in gen_batch:
                    EEG_X_batch, EOG_X_batch, EMG_X_batch, GSR_X_batch,\
                    RSP_X_batch, BLV_X_batch, TMR_X_batch, y_batch = batch
                    sess.run(self._train_op, feed_dict={self.inputs:EEG_X_batch, self.labels:y_batch,
                                                        self.EOG_inputs:EOG_X_batch, self.EMG_inputs:EMG_X_batch,
                                                        self.GSR_inputs:GSR_X_batch, self.RSP_inputs:RSP_X_batch,
                                                        self.BLV_inputs:BLV_X_batch, self.TMR_inputs:TMR_X_batch, 
                                                        self.domain_labels_class_one:domain_labels_class_one,
                                                        self.domain_labels_class_two:domain_labels_class_two,
                                                        self.learning_rate:lr, self._training:True, self.l:l})
                # 打印消息
                if epoch % 1 == 0:
                    total_loss, predictor_loss, feature_loss, label_acc, domain_class_one_acc, domain_class_two_acc \
                    = sess.run([self._total_loss, self._predictor_loss, self._feature_loss, self._label_acc, 
                                self._domain_class_one_acc, self._domain_class_two_acc], 
                                feed_dict={self.inputs:EEG_X_batch, self.labels:y_batch,
                                           self.EOG_inputs:EOG_X_batch, self.EMG_inputs:EMG_X_batch,
                                           self.GSR_inputs:GSR_X_batch, self.RSP_inputs:RSP_X_batch,
                                           self.BLV_inputs:BLV_X_batch, self.TMR_inputs:TMR_X_batch, 
                                           self.domain_labels_class_one:domain_labels_class_one,
                                           self.domain_labels_class_two:domain_labels_class_two})
                    print("{}  Train total_loss: {:.4f}  predictor_loss: {:.4f}  feature loss: {:.4f}".format(epoch, total_loss, predictor_loss, feature_loss))
                    print("training label_acc: {:.2f}%  domain_one_acc: {:.2f}%  domain_two_acc: {:.2f}%".format(label_acc*100, domain_class_one_acc*100, domain_class_two_acc*100))
                    test_acc, test_loss = sess.run([self._label_acc, self._predictor_loss], 
                                                    feed_dict={self.inputs:EEG_X_test, self.labels:y_test,
                                                               self.EOG_inputs:EOG_X_test, self.EMG_inputs:EMG_X_test,
                                                               self.GSR_inputs:GSR_X_test, self.RSP_inputs:RSP_X_test,
                                                               self.BLV_inputs:BLV_X_test, self.TMR_inputs:TMR_X_test})
                    print("Test accuracy: {:.2f}%  test loss: {:.4f}".format(test_acc*100, test_loss))

                    if test_acc > best_acc:
                        best_acc = test_acc
                        self.save("./my_model/"+self._num_+"/train_model.ckpt")
            return self
    
    def save(self, path):
        self._saver.save(self._session, path)
    
    def restore(self):
        path = "./my_model/"+self._num_+"/train_model.ckpt"
        self._saver.restore(self._session, path)

    def predict_proba(self, X):
        EEG_X, EOG_X, EMG_X, GSR_X, RSP_X, BLV_X, TMR_X = X
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self.predictor.eval(feed_dict={self.inputs: EEG_X, self.EOG_inputs:EOG_X,
                                                  self.EMG_inputs:EMG_X, self.GSR_inputs:GSR_X,
                                                  self.RSP_inputs:RSP_X, self.BLV_inputs:BLV_X,
                                                  self.TMR_inputs:TMR_X})
    
    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([self.classes_[class_index] for class_index in class_indices], np.int32).reshape(-1) 
