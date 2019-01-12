# -*- coding:UTF-8 -*-
import tensorflow as tf

def associative_loss(source_feature, target_feature, labels, walker_weight=1.0, visit_weight=1.0):
    '''
    from the paper of Associative Domain Adaptation
    '''
    equality_matrix = tf.equal(tf.reshape(labels, [-1, 1]), labels)
    equality_matrix = tf.cast(equality_matrix, tf.float32)
    p_target = (equality_matrix / tf.reduce_sum(equality_matrix, [1], keepdims=True))
 
    match_ab = tf.matmul(source_feature, target_feature, transpose_b=True, name='match_ab')
    p_ab = tf.nn.softmax(match_ab, name='p_ab')
    p_ba = tf.nn.softmax(tf.transpose(match_ab), name='p_ba')
    p_aba = tf.matmul(p_ab, p_ba, name='p_aba')

    walk_loss = tf.losses.softmax_cross_entropy(p_target, tf.log(1e-8 + p_aba), 
                                                weights=walker_weight, scope='loss_aba')
    visit_probability = tf.reduce_mean(p_ab, [0], keepdims=True, name='visit_prob')
    t_nb = tf.shape(p_ab)[1]
    visit_loss = tf.losses.softmax_cross_entropy(tf.fill([1, t_nb], 1.0 / tf.cast(t_nb, tf.float32)),
                                                 tf.log(1e-8 + visit_probability), weights=visit_weight, 
                                                 scope='loss_visit')
    return walk_loss + visit_loss
