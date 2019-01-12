
################################################################
# 此部分实现 JDDA 网络的构建，特征提取基于 TCN 网络
################################################################
import tensorflow as tf
import numpy as np
from tcn import TCN_Model
from sklearn.exceptions import NotFittedError
from utils import batch_generator, test_for_train_samples
from discriminative_loss import get_center_loss
from associative_loss import associative_loss

class JDDA_Model():
    def __init__(self, 
                 sequence_length, 
                 kernel_size, 
                 num_channels, 
                 dropout,
                 batch_size,
                 in_channels,
                 random_state=None, 
                 domain_loss_param = 20,
                 discrimative_loss_param = 0.0,
                 optimizer_class=tf.train.AdamOptimizer):
        self.sequence_length = sequence_length
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.dropout = dropout
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.random_state = random_state
        self.domain_loss_param = domain_loss_param
        self.discrimative_loss_param = discrimative_loss_param
        self.optimizer_class = optimizer_class
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
        self.labels = tf.placeholder(tf.int32, shape=(self.batch_size), name="labels") # 用于计算域损失
        self.labels_tt = tf.placeholder(tf.int32, shape=(self.batch_size), name="labels_tt") # 用于计算域损失
        self.inputs_t = tf.placeholder(tf.float32, shape=(None, self.sequence_length, self.in_channels), name="inputs_t")
        self.labels_t = tf.placeholder(tf.int32, shape=(None), name="labels_t")
        self._training = tf.placeholder_with_default(False, shape=(), name="training")
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name="learning_rate")

        self.source_model = TCN_Model(self.inputs_s, n_outputs, self.num_channels, self.sequence_length, 
                                      self.kernel_size, self.dropout, is_training=self._training)
        self.target_model = TCN_Model(self.inputs_t, n_outputs, self.num_channels, self.sequence_length, 
                                      self.kernel_size, self.dropout, is_training=self._training, reuse=True)

    def _calculation_loss(self):
        # 源域损失
        self.source_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_s, 
                                                                                         logits=self.source_model.logits))
        # 源域和目标域分布度量
        self.domain_loss = self._calculation_domain_loss(method="MMD")
        # 源域类类间距和类间间距
        self.inter_loss, self.intra_loss, self.centers_update_op = get_center_loss(self.source_model.feature, 
                                                                                   self.labels_s, 0.5, 2)
        self.discriminative_loss = self.inter_loss + self.intra_loss
        self.discriminative_loss=self.discriminative_loss/(2*self.batch_size+2*2)

    def _calculation_domain_loss(self, method):
        if method == "CORAL":
            D_s = self.source_model.feature
            D_t = self.target_model.feature
            I = tf.ones([self.batch_size, 1])
            s_temp = tf.matmul(tf.transpose(tf.matmul(tf.transpose(I), D_s)), 
                               tf.matmul(tf.transpose(I), D_s)) / self.batch_size
            conv_s = (1./(self.batch_size-1))*(tf.matmul(tf.transpose(D_s), D_s) - s_temp)
            t_temp = tf.matmul(tf.transpose(tf.matmul(tf.transpose(I), D_t)), 
                               tf.matmul(tf.transpose(I), D_t)) / self.batch_size
            conv_t = (1./(self.batch_size-1))*(tf.matmul(tf.transpose(D_t), D_t) - t_temp)
            return tf.reduce_mean(tf.square(conv_s - conv_t)) / 4
        if method == "Log_CORAL":
            D_s = self.source_model.feature
            D_t = self.target_model.feature
            I = tf.ones([self.batch_size, 1])
            s_temp = tf.matmul(tf.transpose(tf.matmul(tf.transpose(I), D_s)), 
                               tf.matmul(tf.transpose(I), D_s)) / self.batch_size
            cov_s = (1./(self.batch_size-1))*(tf.matmul(tf.transpose(D_s), D_s) - s_temp)
            t_temp = tf.matmul(tf.transpose(tf.matmul(tf.transpose(I), D_t)), 
                               tf.matmul(tf.transpose(I), D_t)) / self.batch_size
            cov_t = (1./(self.batch_size-1))*(tf.matmul(tf.transpose(D_t), D_t) - t_temp)
            eig_source  = tf.self_adjoint_eig(cov_s)
            eig_target  = tf.self_adjoint_eig(cov_t)
            log_cov_source = tf.matmul(eig_source[1], tf.matmul(tf.diag(tf.log(eig_source[0]+1e-8)), 
                                       eig_source[1], transpose_b=True))
            log_cov_target = tf.matmul(eig_target[1], tf.matmul(tf.diag(tf.log(eig_target[0]+1e-8)), 
                                       eig_target[1], transpose_b=True))
            return tf.reduce_mean(tf.square(tf.subtract(log_cov_source,log_cov_target)))
        if method == "MMD":
            D_s = self.source_model.feature
            D_t = self.target_model.feature
            diff_x = tf.reduce_mean(D_s, axis=0) - tf.reduce_mean(D_t, axis=0)
            domain_loss_x = tf.reduce_sum(tf.square(diff_x)) # 边缘分布距离度量

            source_diff = tf.unsorted_segment_mean(D_s, self.labels, 2)
            #target_diff = tf.unsorted_segment_mean(D_t, tf.argmax(self.target_model.softmax_output, axis=1), 2)
            target_diff = tf.unsorted_segment_mean(D_t, self.labels_tt, 2)
            domain_loss_y = tf.reduce_sum(tf.square(source_diff - target_diff)) # 条件分布距离度量
            domain_loss = domain_loss_x + domain_loss_y
            return domain_loss
        if method == "Associative":
            D_s = self.source_model.feature
            D_t = self.target_model.feature
            return associative_loss(D_s, D_t, self.labels_s, 1.0, 0.2)

    def _build_graph(self, n_outputs):
        self._build_model(n_outputs)
        self._calculation_loss()
        # 构建损失节点
        self.loss = self.source_loss + self.domain_loss_param * self.domain_loss + \
                    self.discrimative_loss_param * self.discriminative_loss
        # 构建训练节点
        var_all = tf.trainable_variables()
        with tf.control_dependencies([self.centers_update_op]):
            self.train_op = self.optimizer_class(self.learning_rate).minimize(self.loss, var_list=var_all)
        self.train_op_source_only = self.optimizer_class(self.learning_rate).minimize(self.source_loss, var_list=var_all)
        # 构建评估节点
        correct_source = tf.nn.in_top_k(self.source_model.softmax_output, self.labels_s, 1)
        self.s_predictor = tf.reduce_mean(tf.cast(correct_source, tf.float32))
        correct_target = tf.nn.in_top_k(self.target_model.softmax_output, self.labels_t, 1)
        self.t_predictor = tf.reduce_mean(tf.cast(correct_target, tf.float32))
        self.predictor = self.target_model.softmax_output
        # 构建全局初始化节点
        init = tf.global_variables_initializer()
        # 构建模型保存节点
        saver = tf.train.Saver()
    
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
            X_test_train, y_test_train = test_for_train_samples(X_test, y_test)
            for epoch in range(epochs):
                lr = 0.001
                seed += 1
                gen_batch = batch_generator(X, y, X_test_train, y_test_train, self.batch_size, seed)
                for batch in gen_batch:
                    X_batch_s, y_batch_s, X_batch_t, y_batch_t = batch
                    if epoch <= -1:
                        sess.run(self.train_op_source_only, feed_dict={self.inputs_s:X_batch_s, self.labels_s:y_batch_s,
                                                                       self.learning_rate:lr, self._training:True})
                    else:
                        sess.run(self.train_op, feed_dict={self.inputs_s:X_batch_s, self.labels_s:y_batch_s, 
                                                           self.labels:y_batch_s, self.labels_tt:y_batch_t,
                                                           self.inputs_t:X_batch_t, self.labels_t:y_batch_t,
                                                           self.learning_rate:lr, self._training:True})
                # 打印消息
                if epoch % 1 == 0:
                    total_loss, source_loss, domain_loss, intra_loss, inter_loss, train_acc = sess.run([self.loss, 
                                                       self.source_loss, self.domain_loss, self.intra_loss, 
                                                       self.inter_loss, self.s_predictor], 
                                                       feed_dict={self.inputs_s:X_batch_s, self.labels_s:y_batch_s, 
                                                                  self.labels:y_batch_s, self.labels_tt:y_batch_t,
                                                                  self.inputs_t:X_batch_t, self.labels_t:y_batch_t})
                    print("{}  Training total_loss: {:.4f}  source_loss: {:.4f}  domain_loss: {:.6f}  intra_loss: {:.4f}  inter_loss: {:.4f}".format(epoch, total_loss, source_loss, domain_loss, intra_loss, inter_loss))
                    print("training accuracy: {:.2f}%".format(train_acc*100))

                    test_acc, test_loss = sess.run([self.t_predictor, self.source_loss], 
                                                    feed_dict={self.inputs_t:X_test, self.labels_t:y_test,
                                                               self.inputs_s:X_test, self.labels_s:y_test})
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
            return self.predictor.eval(feed_dict={self.inputs_t: X})
    
    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([self.classes_[class_index] for class_index in class_indices], np.int32).reshape(-1)
