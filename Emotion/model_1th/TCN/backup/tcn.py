# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
# 并不是所有的方法都能用 arg_scope 设置默认参数
# 只有用 add_arg_scope 修饰过的方法才能使用arg_scope.

# 构建 He initializer 节点
he_init = tf.contrib.layers.variance_scaling_initializer()
# 构建 random_normal_initializer 初始化节点
random_normal_init = tf.random_normal_initializer(0, 0.01)
def get_name(layer_name, counters):
    '''
    用来跟踪网络层的名字
    参数：
      layer_name：一个字符串
      counters: 一个字典
    '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

def temporal_padding(x, padding=(1,1)):
    '''用来填充（填充 0）, 填充一个 3D tensor 的中间维度'''
    assert len(padding) == 2
    pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
    return tf.pad(x, paddings=pattern)

@add_arg_scope
def weight_norm_convolution1D(x, num_filters, dilation_rate, filter_size=3, stride=[1],pad="VALID", 
                              init_scale=1., init=False, gated=False, counters={}, reuse=False):
    '''带有权重归一化的扩张卷积'''
    name = get_name('weight_norm_conv1d', counters)
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        if init:
            print("Initializing weight norm")
            # data based initialization of parameters
            V = tf.get_variable('V', [filter_size, int(x.get_shape()[-1]), num_filters], tf.float32, 
                                he_init, trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1])
            # pad x
            left_pad = dilation_rate * (filter_size - 1)
            x = temporal_padding(x, (left_pad, 0))
            x_init = tf.nn.convolution(x, V_norm, pad, stride, [dilation_rate])
            m_init, v_init = tf.nn.moments(x_init, [0, 1])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init, trainable=True)
            b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init*scale_init, trainable=True)
            x_init = tf.reshape(scale_init, [1,1,num_filters]) * (x_init - tf.reshape(m_init, [1,1,num_filters]))
            # apply nonlinearity
            x_init = tf.nn.relu(x_init)
            return x_init
        else:
            if gated:
                num_filters = num_filters * 2
            # size of V is L, Cin, Cout
            V = tf.get_variable('V', [filter_size, int(x.get_shape()[-1]), num_filters], tf.float32, 
                                he_init, trainable=True)
            g = tf.get_variable('g', shape=[num_filters], dtype=tf.float32, 
                                initializer=tf.constant_initializer(1.), trainable=True)
            b = tf.get_variable('b', shape=[num_filters], dtype=tf.float32, 
                                initializer=None, trainable=True)
            # 使用权重初始化
            W = tf.reshape(g, [1,1,num_filters]) * tf.nn.l2_normalize(V, [0, 1])
            # 填充 x 用于因果卷积
            left_pad = dilation_rate * (filter_size - 1)
            x = temporal_padding(x, (left_pad, 0))
            # 计算卷积层输出
            x = tf.nn.bias_add(tf.nn.convolution(x, W, pad, stride, [dilation_rate]), b)
            if gated:
                split0, split1 = tf.split(x, num_or_size_splits=2, axis=2)
                split1 = tf.sigmoid(split1)
                x = tf.multiply(split0, split1)
            else:
                x = tf.nn.relu(x)
            print(x.get_shape())
            return x

def temporal_block(input_layer, out_channels, filter_size, stride, dilation_rate, counters, 
                   dropout, init=False, use_highway=False, gated=False):
    '''TCN 中残差块'''
    keep_prob = 1.0 - dropout
    in_channels = input_layer.get_shape()[-1]
    name = get_name('temporal_block', counters)
    with tf.variable_scope(name):
        conv1 = weight_norm_convolution1D(input_layer, out_channels, dilation_rate, filter_size, 
                                          [stride], counters=counters, init=init, gated=gated)
        noise_shape = (tf.shape(conv1)[0], tf.constant(1), tf.shape(conv1)[2])
        out1 = tf.nn.dropout(conv1, keep_prob, noise_shape)

        conv2 = weight_norm_convolution1D(out1, out_channels, dilation_rate, filter_size, [stride], 
                                          counters=counters, init=init, gated=gated)
        out2 = tf.nn.dropout(conv2, keep_prob, noise_shape)
        residual = None
        if in_channels != out_channels:
            W_h = tf.get_variable('W_h', [1, int(input_layer.get_shape()[-1]), out_channels], 
                                  tf.float32, he_init, trainable=True)
            b_h = tf.get_variable('b_h', shape=[out_channels], dtype=tf.float32, initializer=None, trainable=True)
            residual = tf.nn.bias_add(tf.nn.convolution(input_layer, W_h, 'SAME'), b_h)
        else:
            print("No residual convolution")
        res = input_layer if residual is None else residual
        return tf.nn.relu(out2 + res)

def temporal_ConvNet(input_layer, num_channels, sequence_length, kernel_size=3, 
                     dropout=tf.constant(0.0, dtype=tf.float32), init=False, use_gated=False):
    num_levels = len(num_channels)
    counters = {}
    for i in range(num_levels):
        print(i)
        dilation_size = 2 ** i
        out_channels = num_channels[i]
        input_layer = temporal_block(input_layer, out_channels, kernel_size, stride=1, dilation_rate=dilation_size,
                                     counters=counters, dropout=dropout, init=init, gated=use_gated)
    return input_layer

def TCN(input_layer, output_size, num_channels, sequence_length, kernel_size, dropout):
    tcn = temporal_ConvNet(input_layer=input_layer, num_channels=num_channels, sequence_length=sequence_length, 
                           kernel_size=kernel_size, dropout=dropout)
    tcn_flat = tf.contrib.layers.flatten(tcn)
    linear = tf.contrib.layers.fully_connected(tcn_flat, output_size, activation_fn=None)
    return linear
