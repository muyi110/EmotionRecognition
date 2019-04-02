# -*- coding:UTF-8 -*-
########################################################################
# 通过自编码器获取外周生理信号特征
########################################################################
import os
import sys
sys.path.append("../../common/") # 将其他模块路径添加到系统搜索路径
from read_peripheral_physiological_signal_data import get_data
import numpy as np
import tensorflow as tf

class AutoEncoder():
    def __init__(self,
                 n_inputs=128,
                 n_hidden1=50,
                 n_hidden2=10,
                 n_hidden3=50,
                 n_hidden4=128,
                 learning_rate=0.0001,
                 optimizer_class=tf.train.AdamOptimizer):
        self.n_inputs = n_inputs
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.n_hidden4 = n_hidden4
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self._session = None
        self.initializer = tf.contrib.layers.variance_scaling_initializer()

    def _build_graph(self, noise_level=0.5):
        self.X = tf.placeholder(tf.float32, shape=[None, self.n_inputs], name="X")
        self.X_noisy = self.X + tf.random_normal(tf.shape(self.X)) * noise_level
        # 自编码器网络结构搭建
        hidden1 = tf.layers.dense(self.X_noisy, self.n_hidden1, activation=tf.nn.elu, 
                                  kernel_initializer=self.initializer, name="hidden1")
        hidden2 = tf.layers.dense(hidden1, self.n_hidden2, activation=tf.nn.elu, 
                                  kernel_initializer=self.initializer, name="hidden2")
        hidden3 = tf.layers.dense(hidden2, self.n_hidden3, activation=tf.nn.elu, 
                                  kernel_initializer=self.initializer, name="hidden3")
        outputs = tf.layers.dense(hidden3, self.n_hidden4, activation=None, 
                                  kernel_initializer=self.initializer, name="outputs")
        # 构建损失节点
        self.reconstruction_loss = tf.reduce_mean(tf.square(outputs - self.X))
        # 构建优化器节点
        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        self.training_op = optimizer.minimize(self.reconstruction_loss)
        # 构建全局初始化节点和模型保存节点
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        # 构建外周生理信号特征节点
        self.features = hidden2

    def fit(self, X, epochs=301, batch_size=60, noise_level=0.5):
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(noise_level)
        # 开始训练阶段
        if self._session:
            self._session.close()
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            sess.run(self.init)
            for epoch in range(epochs):
                # 获取 batch 数据
                batches = get_batch_data(X, batch_size)
                for batch in batches:
                    sess.run(self.training_op, feed_dict={self.X:batch})
                # 打印信息
                if epoch % 10 == 0:
                    batch_loss = sess.run(self.reconstruction_loss, feed_dict={self.X:batch})
                    print("{}  loss: {:.6f}".format(epoch, batch_loss))
                    self.saver.save(sess, "./model_params/my_model_autoencoder.ckpt")
            return self
        
def get_batch_data(X, batch_size):
    assert(X.shape[1] == 128 and len(X.shape) == 2)
    num = X.shape[0]
    batch_numbers = num // batch_size
    permutation = list(np.random.permutation(num))
    shuffled_X = X[permutation]
    assert(shuffled_X.shape == (num, 128))
    batches = []
    for i in range(batch_numbers):
        batch = shuffled_X[i*batch_size:(i+1)*batch_size, :]
        assert(batch.shape == (batch_size, 128))
        batches.append(batch)
    assert(len(batches) == 1*40)
    return batches

def _save_samples(datas_result, labels, people_list, classify_object_name, channel):
    if classify_object_name == 0:
        class_name = "valence_peripheral"
    elif classify_object_name == 1:
        class_name = "arousal_peripheral"
    # 针对单独一个人情况
    if len(people_list) == 1:
        print("save samples start: ")
        if not os.path.isdir(os.path.join("../common/peripheral_features/"+class_name, "s"+str(people_list[0]))):
            os.makedirs(os.path.join("../common/peripheral_features/"+class_name, "s"+str(people_list[0])))
        np.save("./peripheral_features/"+class_name+"/s"+str(people_list[0])+"/datas_"+str(channel), datas_result)
        np.save("./peripheral_features/"+class_name+"/s"+str(people_list[0])+"/labels_"+str(channel), labels)

if __name__ == "__main__":
    people_list = list(range(0, 32))
    trial_list = list(range(0, 40))
    datas, labels = get_data(people_list, trial_list, classify_object_name=0)
    datas = np.array(datas)
    labels = np.array(labels)
    assert(datas.shape == (32*40, 8, 7680))
    assert(labels.shape == (32*40,))
    # 以秒为单位，将样本划分为多个时间片段
    for people in range(32):
        tf.reset_default_graph()
        peripheral_data = datas[people*40:(people+1)*40, :, :]
        assert(peripheral_data.shape == (40, 8, 7680))
        for channel in range(8): # 8个外周生理信号通道数据
            samples = []
            signal_data = peripheral_data[:, channel, :]
            assert(signal_data.shape == (40, 7680))
            for i in range(60):
                data_1s = signal_data[:, i*128:(i+1)*128]
                assert(data_1s.shape == (40, 128))
                max_value = data_1s.max(axis=1, keepdims=True)
                assert(max_value.shape == (40, 1))
                min_value = data_1s.min(axis=1, keepdims=True)
                data_normalization = (data_1s - min_value) / (max_value - min_value + 1e-8)
                assert(data_normalization.shape == (40, 128))
                samples.append(data_normalization)
            samples = np.array(samples).reshape(-1, 128)
            assert(samples.shape == (40*60, 128))
            autoencoder = AutoEncoder()
            autoencoder.fit(samples)
            # 开始保存特征
            with autoencoder._session.as_default() as sess:
                autoencoder.saver.restore(sess, "./model_params/my_model_autoencoder.ckpt")
                features, loss  = sess.run([autoencoder.features, autoencoder.reconstruction_loss], 
                                           feed_dict={autoencoder.X:samples})
                print("loss: ", loss)
                assert(features.shape == (2400, 10))
                # 将特征转为 shape=(40, 10, 60)
                features_result = []
                for p in range(40):
                    temp_list = []
                    for q in range(60):
                        temp_list.append(features[p+40*q:(p+1)+40*q, :].reshape(10, 1))
                    features_result.append(np.hstack(temp_list))
                features_result = np.array(features_result)
                assert(features_result.shape == (40, 10, 60))
                labels_one_people = labels[people*40:(people+1)*40]
                assert(labels_one_people.shape == (40, ))
                _save_samples(features_result, labels_one_people, [people], 0, channel)
