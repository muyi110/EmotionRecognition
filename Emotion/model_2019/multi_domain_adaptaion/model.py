# -*- coding:UTF-8 -*-
import tensorflow as tf
import numpy as np
from tcn import TCN
from utils import batch_generator
from gradient_reversal import gradient_reversal
from sklearn.exceptions import NotFittedError

class MDANN():
    def __init__(self, 
                 sequence_length,
                 kernel_size,
                 num_channels,
                 dropout,
                 batch_size,
                 in_channels,
                 number_of_class_one,
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
        self._session = None

    def _build_model(self, n_outputs):
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)
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
        with tf.variable_scope("feature_extractor"):
            # features 的 shape=(samples, sequence_length, features)
            features = TCN(self.inputs, n_outputs, self.num_channels, self.sequence_length, 
                           self.kernel_size, self.dropout, is_training=self._training)
            self.feature = features[:, -1, :]
        # softmax 用于情绪的二分类
        with tf.variable_scope("label_predictor"):
            logits = tf.contrib.layers.fully_connected(self.feature, n_outputs, activation_fn=None)
            self.predictor = tf.nn.softmax(logits, name="predictor")
            self.predictor_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
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

    def _build_graph(self, n_outputs):
        self._build_model(n_outputs)
        # 构建损失节点（情绪分类损失， 两个域分类损失， 最后一层特征约束损失：同一情绪类别差异小， 不同情绪类别差异大）
        predictor_loss = tf.reduce_mean(self.predictor_loss) # 情绪分类损失
        domain_loss = tf.reduce_mean(self.domain_loss_class_one) + tf.reduce_mean(self.domain_loss_class_two) # 域损失
        feature_loss = self._calculate_feature_loss() # 特征损失
        # (valence 准确率 75.4% 的参数是 2.2 和 0.2)
        total_loss = predictor_loss + domain_loss*3.2 + feature_loss * 0.1
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
            #for j in range(i+1, self.number):
            #    L1_diff_class_one += tf.reduce_sum(tf.square(feature_class_one[i] - feature_class_one[j]))
        #L1_diff_class_one = L1_diff_class_one / (self.number*(self.number-1) / 2)
        L1_diff_class_one = L1_diff_class_one / self.number
        L1_diff_class_two = 0.0
        for m in range(32-self.number):
            L1_diff_class_two += tf.reduce_sum(tf.square(feature_class_two[i]-tf.reduce_mean(self.feature[self.number:],0)))
            #for n in range(m+1, 32-self.number):
            #    L1_diff_class_two += tf.reduce_sum(tf.square(feature_class_two[m] - feature_class_two[n]))
        #L1_diff_class_two = L1_diff_class_two / ((32-self.number)*(32-self.number-1) / 2)
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
                    X_batch, y_batch = batch
                    sess.run(self._train_op, feed_dict={self.inputs:X_batch, self.labels:y_batch, 
                                                        self.domain_labels_class_one:domain_labels_class_one,
                                                        self.domain_labels_class_two:domain_labels_class_two,
                                                        self.learning_rate:lr, self._training:True, self.l:l})
                # 打印消息
                if epoch % 1 == 0:
                    total_loss, predictor_loss, feature_loss, label_acc, domain_class_one_acc, domain_class_two_acc \
                    = sess.run([self._total_loss, self._predictor_loss, self._feature_loss, self._label_acc, 
                                self._domain_class_one_acc, self._domain_class_two_acc], 
                                feed_dict={self.inputs:X_batch, self.labels:y_batch, 
                                           self.domain_labels_class_one:domain_labels_class_one,
                                           self.domain_labels_class_two:domain_labels_class_two})
                    print("{}  Train total_loss: {:.4f}  predictor_loss: {:.4f}  feature loss: {:.4f}".format(epoch, total_loss, predictor_loss, feature_loss))
                    print("training label_acc: {:.2f}%  domain_one_acc: {:.2f}%  domain_two_acc: {:.2f}%".format(label_acc*100, domain_class_one_acc*100, domain_class_two_acc*100))
                    test_acc, test_loss = sess.run([self._label_acc, self._predictor_loss], 
                                                    feed_dict={self.inputs:X_test, self.labels:y_test})
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
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self.predictor.eval(feed_dict={self.inputs: X})
    
    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([self.classes_[class_index] for class_index in class_indices], np.int32).reshape(-1) 
