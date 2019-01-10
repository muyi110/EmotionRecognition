# -*- coding:UTF-8 -*-
######################################################################
# DeepJDOT 网络实现，特征提取利用 TCN 网络
######################################################################
import numpy as np
import ot
from scipy.spatial.distance import cdist
import tensorflow as tf
from sklearn.exceptions import NotFittedError
from tcn import TCN
from utils import batch_generator

class DeepJDOT():
    def __init__(self,
                 sequence_length,
                 kernel_size,
                 num_channels,
                 dropout,
                 batch_size,
                 in_channels,
                 jdot_alpha=20,
                 jdot_lambda=1,
                 learning_rate=0.001,
                 lr_decay=True,
                 optimizer_class=tf.train.AdamOptimizer):
        self.sequence_length=sequence_length
        self.kernel_size=kernel_size
        self.num_channels=num_channels
        self.dropout=dropout
        self.batch_size=batch_size
        self.in_channels=in_channels
        self.optimizer_class=optimizer_class
        self.jdot_alpha=jdot_alpha
        self.jdot_lambda=jdot_lambda
        self.gamma=np.zeros(shape=(self.batch_size//2, self.batch_size//2)) # 零初始化 gamma (coupling in OT)
        self.lr=learning_rate
        self.lr_decay=lr_decay
        self._session = None

    def _build_model(self, n_outputs):
        '''
        n_outputs: 输出类别数
        '''
        tf.set_random_seed(42)
        np.random.seed(42)
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.sequence_length, self.in_channels), name="inputs")
        self.labels = tf.placeholder(tf.int32, shape=(None), name="labels")
        self._training = tf.placeholder_with_default(False, shape=(), name="is_training")
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")
        self.gamma_tf = tf.placeholder(tf.float32, shape=(self.batch_size//2, self.batch_size//2), name="gamma")
        # 利用 TCN 模型提取特征
        with tf.variable_scope("feature_extractor"):
            # feature.shape = (samples, seq_length, features)
            features = TCN(self.inputs, n_outputs, self.num_channels, self.sequence_length, 
                               self.kernel_size, self.dropout, is_training=self._training)
            # 取最后一个输出作为最终特征
            self.feature = features[:, -1, :]
        # softmax 用于分类预测
        with tf.variable_scope("label_predictor"):
            # temp = tf.contrib.layers.fully_connected(self.feature, 10, activation_fn=tf.nn.relu)
            logits = tf.contrib.layers.fully_connected(self.feature, n_outputs, activation_fn=None)
            self.predictor = tf.nn.softmax(logits, name="predictor")
            # 下面开始计算源域损失(分类损失)
            source_logits = tf.slice(logits, [0, 0], [self.batch_size//2, -1])
            source_labels = tf.slice(self.labels, [0], [self.batch_size//2])
            self.source_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=source_logits, 
                                                                                             labels=source_labels))
            # 评估模型的时候用到的损失
            self.pre_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, 
                                                                                          labels=self.labels))
            # 下面开始计算目标域损失(cross-entropy loss)
            target_predictor = tf.slice(self.predictor, [self.batch_size//2, 0], [self.batch_size//2, -1])
            source_labels_one_hot = tf.one_hot(source_labels, 2)
            target_loss = tf.multiply(tf.matmul(source_labels_one_hot, tf.transpose(target_predictor)), -1)
            self.target_loss = tf.reduce_sum(tf.multiply((self.gamma_tf * target_loss), self.jdot_lambda))
            # # 下面开始计算目标损失(hinge loss)
            # target_predictor = tf.slice(self.predictor, [self.batch_size//2, 0], [self.batch_size//2, -1])
            # # 将预测的概率转为对应的类别标签
            # target_pre_labels = tf.argmax(target_predictor, axis=1)
            # target_predictor = tf.cast(tf.reshape(target_pre_labels, (1, -1)) * 2 - 1, tf.int32)
            # source_labels = tf.cast(tf.reshape(source_labels, (-1, 1)) * 2 -1, tf.int32)
            # target_loss = tf.cast(tf.maximum(0, 1-tf.matmul(source_labels, target_predictor)), tf.float32)
            # self.target_loss = tf.reduce_sum(tf.multiply((self.gamma_tf * target_loss), self.jdot_lambda))
            # 下面开始计算特征损失(feature allignment loss)
            gs = tf.slice(self.feature, [0, 0], [self.batch_size//2, -1]) # 源域的特征
            gt = tf.slice(self.feature, [self.batch_size//2, 0], [self.batch_size//2, -1]) # 目标域的特征
            dist1 = tf.reshape(tf.reduce_sum(tf.square(gs), axis=1), (-1, 1))
            dist2 = dist1 + tf.reshape(tf.reduce_sum(tf.square(gt), axis=1), (1, -1))
            gdist = dist2 - 2.0*tf.matmul(gs, tf.transpose(gt))
            self.allign_loss = self.jdot_alpha * tf.reduce_sum(self.gamma_tf * gdist)
       
            self._target_loss, self._gdist = target_loss, gdist

    def _build_graph(self, n_outputs):
        self._build_model(n_outputs)
        # 构建损失节点
        total_loss = self.target_loss + self.allign_loss + self.source_loss*2.0
        c_loss = self.target_loss*2 + self.source_loss*1.0
        f_loss = self.allign_loss
        pre_loss = self.pre_loss
        # 构建训练节点
        train_vars_f = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="feature_extractor")
        train_vars_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="label_predictor")
        train_op = self.optimizer_class(self.learning_rate).minimize(total_loss, var_list=train_vars_c+train_vars_f)
        train_op_c = self.optimizer_class(self.learning_rate).minimize(c_loss, var_list=train_vars_c+train_vars_f)
        train_op_f = self.optimizer_class(self.learning_rate).minimize(f_loss, var_list=train_vars_f)
        source_train_op = self.optimizer_class(learning_rate=self.learning_rate).minimize(pre_loss)
        # 构建评估节点
        correct_label_predictor = tf.nn.in_top_k(self.predictor, self.labels, 1)
        label_acc = tf.reduce_mean(tf.cast(correct_label_predictor, tf.float32))
        # 构建全局初始化节点
        init = tf.global_variables_initializer()
        # 构建模型保存节点
        saver = tf.train.Saver()

        self._pre_loss, self._train_op, self._label_acc, self._total_loss = pre_loss, train_op, label_acc, total_loss
        self._source_train_op = source_train_op
        self._train_op_c, self._train_op_f = train_op_c, train_op_f
        self._init, self._saver = init, saver
    
    def close_session(self):
        if self._session:
            self._session.close()

    def fit(self, X, y, num_steps, X_test, y_test, outputs, people_num=None):
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
            gen_source_batch = batch_generator([X, y], self.batch_size//2)
            gen_target_batch = batch_generator([np.vstack([X_test, X_test, X_test, X_test]), 
                                                np.hstack([y_test, y_test, y_test, y_test])], self.batch_size//2)
            for step in range(num_steps):
                if step < 0:
                    X_batch, y_batch = next(gen_source_batch)
                    sess.run(self._source_train_op, feed_dict={self.inputs:X_batch, self.labels:y_batch, 
                                                           self.learning_rate:self.lr, self._training:True})
                else:
                    # 学习率衰减
                    if self.lr_decay and step % 10000 == 0 and step !=0:
                        self.lr *= 0.1
                    X0, y0 = next(gen_source_batch)
                    X1, y1 = next(gen_target_batch)
                    X_batch = np.vstack([X0, X1])
                    y_batch = np.hstack([y0, y1])
                    # 固定网络的参数，更新 coupling in OT
                    C1, C0 = sess.run([self._target_loss, self._gdist], 
                                       feed_dict={self.inputs:X_batch, self.labels:y_batch})
                    C = (self.jdot_alpha * C0) + (C1 * self.jdot_lambda)
                    # 优化 copuling
                    gamma = ot.emd(ot.unif(self.batch_size//2), ot.unif(self.batch_size//2), C)
                    # 更新 gamma 参数 (coupling in OT)
                    self.gamma = gamma
                    # 固定 coupling 更新网络的参数
                    sess.run([self._train_op_c], feed_dict={self.inputs:X_batch, self.labels:y_batch, 
                                                            self.learning_rate:self.lr,
                                                            self.gamma_tf:self.gamma, 
                                                            self._training:True})
                    #sess.run([self._train_op_f], feed_dict={self.inputs:X_batch, self.labels:y_batch, 
                    #                                        self.learning_rate:self.lr,
                    #                                        self.gamma_tf:self.gamma, 
                    #                                        self._training:True})
                # 打印信息
                if step % 50 == 0 and step != 0:
                    train_loss, train_acc = sess.run([self._pre_loss, self._label_acc], 
                                                      feed_dict={self.inputs:X, self.labels:y})
                    print("{}\ttraining loss: {:.4f}\t|  trainning accuracy: {:.2f}%".format(step, 
                          train_loss, train_acc*100))
                    test_loss, test_acc = sess.run([self._pre_loss, self._label_acc], 
                                                      feed_dict={self.inputs:X_test, self.labels:y_test})
                    print("test loss: {:.4f}\t|  test accuracy: {:.2f}%".format(test_loss, test_acc*100))
                    if test_acc >= best_acc:
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
