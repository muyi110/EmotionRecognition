# -*- coding:UTF-8 -*-
import tensorflow as tf

def get_center_loss(features, labels, alpha, num_class):
    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_class, len_features], dtype=tf.float32, 
                              initializer=tf.constant_initializer(0), trainable=False) 
    labels = tf.reshape(labels, [-1])
    centers0 = tf.unsorted_segment_mean(features, labels, num_class)
    edge_weight = tf.ones(shape=(num_class, num_class)) - tf.eye(num_class)
    margin = tf.constant(100, dtype="float32")
    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    center_pairwise_dist = tf.transpose(norm(tf.expand_dims(centers0, 2) - tf.transpose(centers0)))
    loss_0 = tf.reduce_sum(tf.multiply(tf.maximum(0.0, margin-center_pairwise_dist), edge_weight))

    # 根据label, 获取 mini-batch 中每一个样本对应中心值
    centers_batch = tf.gather(centers, labels)
    diff = centers_batch - features
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    loss_1 = tf.nn.l2_loss(features - centers_batch)
    centers_update_op= tf.scatter_sub(centers, labels, diff)
    return loss_0, loss_1, centers_update_op
