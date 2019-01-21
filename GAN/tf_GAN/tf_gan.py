#! /usr/bin/env python3
# -*- coding:UTF-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def variable_init(size):
    '''
    权重初始化
    '''
    in_dim = size[0]
    w_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=w_stddev)

X = tf.placeholder(tf.float32, shape=[None, 784])
# 定义判别器的权重和偏置
D_W1 = tf.Variable(variable_init([784, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(variable_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]
# 定义生成器的权重，偏置及输入
Z = tf.placeholder(tf.float32, shape=[None, 100])
G_W1 = tf.Variable(variable_init([100, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(variable_init([128, 784]))
G_b2 = tf.Variable(tf.zeros(shape=[784]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

def sample_Z(m, n):
    '''
    随机从均匀分布采样，作为生成器的输入
    '''
    return np.random.uniform(-1., 1., size=[m, n])

def generator(z):
    '''
    生成器
    '''
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob

def discriminator(x):
    '''
    判别器
    '''
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit

def plot(samples):
    '''
    用于生成输出的图片
    '''
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap="Greys_r")
    return fig

# 获取生成样本
G_sample = generator(Z)
# 将生成样本和真实样本输入判别器，获取输出
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)
# 构建损失函数（采用交叉熵）
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, 
                                                                     labels = tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, 
                                                                     labels = tf.zeros_like(D_logit_real)))
D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                labels=tf.ones_like(D_logit_fake)))

# 定义训练 OP
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
# 获取样本
mb_size = 128
Z_dim = 100
mnist = input_data.read_data_sets("./data/MNIST/", one_hot=True)
# 构建会话，并开始训练
sess = tf.Session()
sess.run(tf.global_variables_initializer())
if not os.path.exists('out/'):
    os.makedirs("out/")

i = 0
for it in range(200001):
    if it % 2000 == 0:
        samples = sess.run(G_sample, feed_dict={Z:sample_Z(16, Z_dim)})
        fig = plot(samples)
        plt.savefig("out/{}.png".format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
    X_mb, _ = mnist.train.next_batch(mb_size)
    # 先训练 k 次判断器，后训练 1 次生成器 (按照论文里，这里 k=1)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X:X_mb, Z:sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z:sample_Z(mb_size, Z_dim)})
    
    if it % 2000 == 0:
        print('{}  D_loss: {:.4f}  G_loss: {:.4f}'.format(it, D_loss_curr, G_loss_curr))
