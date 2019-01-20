# -*- coding:UTF-8 -*-
import tensorflow as tf
from model.build_gen import Generator, Classifier
from datasets.dataset_read import dataset_read

class Solver():
    def __init__(self, batch_size=64, source="uspsa", target="mnist", learning_rate=0.0002, num_k=4):
        self.batch_size = batch_size
        self.source = source
        self.target = target
        self.num_k = num_k
        self.learning_rate = learning_rate
        
        self.X = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name="X")
        #self.y = tf.place
        self.is_training = tf.placeholder_with_default(False, shape=(), name="is_training")

    def _build_graph(self):
        with tf.variable_scope("feature"):
            self.G = Generator(self.X, self.is_training)
        with tf.variable_scope("classifier1"):
            self.C1 = Classifier(self.G.feature)
        with tf.variable_scope("classifier2"):
            self.C2 = Classifier()
