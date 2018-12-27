# -*- coding:UTF-8 -*-
#######################################################################
# 此模块实现域对抗网络（DANN）中的梯度翻转层（gradient reversal layer）
#######################################################################
import tensorflow as tf
from tensorflow.python.framework import ops

class GradientReversalBuilder():
    def __init__(self):
        self._num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "GradientReversal%d" % self._num_calls
        @ops.RegisterGradient(grad_name)
        def _gradients_reversal(op, grad):
            return [tf.negative(grad) * l]
        
        g = tf.get_default_graph()
        # gradient_override_map({op_A_type: op_B_type}) 是用 op_B 的反向传播机制代替 op_A 的反向传播机制，
        # 然而保留 op_A 的前向传播机制 
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self._num_calls += 1
        return y

gradient_reversal = GradientReversalBuilder()
