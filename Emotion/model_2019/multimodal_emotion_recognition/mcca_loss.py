# -*- coding: UTF-8 -*-
################################################################
# 多域典型相关分析损失函数
################################################################
import tensorflow as tf

def mcca_loss(N, F, batch_size, gamma = 0.2, lambda_ = 0.0001):
    '''
    N--一个元组，包括每一个模态的特征个数
    F--输入的特征，samples*features, 其中特征数为各个模态的特征级联
    '''
    modality_num = len(N)
    modality_range = [(N[0]*i, N[0]*(i+1)) for i in range(modality_num)]
    m = batch_size
    xbar = tf.transpose(F) - (1.0/m)*tf.matmul(tf.transpose(F), tf.ones([m, m])) # 中心化
    x_Rw = (1.0 / (m-1))*tf.matmul(xbar, tf.transpose(xbar)) # 协方差矩阵
    Rw_ = tf.zeros([N[0], N[0]])
    Xsum = tf.zeros([N[0], m])
    for i, j in modality_range:
        Rw_ = Rw_ + x_Rw[i:j, i:j]
        Xsum = Xsum + tf.transpose(F)[i:j, :]
    Xmean = Xsum / modality_num
    Xmean_bar = Xmean - (1.0 / m) * tf.matmul(Xmean, tf.ones([m, m]))
    Rt_ = ((modality_num*modality_num*1.0))/(m-1) * tf.matmul(Xmean_bar, tf.transpose(Xmean_bar))
    Rb_ = (Rt_ - Rw_) / (modality_num-1)
    Rw_reg_ = ((1-gamma)*Rw_) + (gamma*(tf.reduce_mean(tf.diag_part(Rw_))))*tf.eye(N[0])
    L = tf.linalg.cholesky(Rw_reg_)
    Linv = tf.matrix_inverse(L)
    C_ = tf.matmul(Linv, Rb_)
    C = tf.matmul(C_, tf.transpose(Linv))
    C_eigval, C_eigvec = tf.linalg.eigh(C)
    index_= tf.contrib.framework.argsort(C_eigval)[::-1] # 翻转读取(沿着行方向)
    W_ = tf.matmul(tf.transpose(Linv), C_eigvec)
    W_ = tf.transpose(W_)
    W_ = tf.transpose(tf.gather(W_, index_))
    d_ = tf.diag(1.0 / tf.sqrt(tf.reduce_sum(W_*W_, axis=0)))
    W_ = tf.matmul(W_, d_)
    ISC = tf.diag(tf.matmul(tf.matmul(tf.transpose(W_), Rb_), W_)) / tf.diag(tf.matmul(tf.matmul(tf.transpose(W_), Rw_), W_))
    corr = tf.sqrt(tf.reduce_sum(ISC*ISC))
    return -1*tf.reduce_mean(ISC)
