#! /usr/bin/env python3
# -*- coding:UTF-8 -*-
import tensorflow as tf
tf.enable_eager_execution()
import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
from IPython import display

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# 归一化到 [-1, 1]
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = tf.keras.layers.Dense(7*7*64, use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        
        self.conv1 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='SAME', use_bias=False)
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='SAME', use_bias=False)
        self.batchnorm3 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="SAME", use_bias=False)
    def call(self, x, training=True):
        x = self.fc1(x)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.relu(x)
        x = tf.reshape(x, shape=(-1, 7, 7, 64))

        x = self.conv1(x)
        x = self.batchnorm2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.batchnorm3(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv3(x)
        x = tf.nn.tanh(x)
        return x

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1)
    def call(self, x, training=True):
        x = tf.nn.leaky_relu(self.conv1(x))
        x = self.dropout(x, training=training)
        x = tf.nn.leaky_relu(self.conv2(x))
        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

generator = Generator()
discriminator = Discriminator()
# 下面两行用来提升性能
generator.call = tf.contrib.eager.defun(generator.call)
discriminator.call = tf.contrib.eager.defun(discriminator.call)

# 构建损失函数和优化器
def discriminator_loss(real_output, generated_output):
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)
    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output),
                                                     logits=generated_output)
    total_loss = real_loss + generated_loss
    return total_loss

def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(generated_output), logits=generated_output)

discriminator_optimizer = tf.train.AdamOptimizer(1e-4)
generator_optimizer = tf.train.AdamOptimizer(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, 
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
# 开始训练
EPOCHS = 150
noise_dim = 100
num_examples_to_generate = 16
random_vector_for_generation = tf.random_normal([num_examples_to_generate, noise_dim])

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0]*127.5+127.5, cmap='gray')
        plt.axis('off')
    plt.savefig('./images/image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()

def train(dataset, epochs, noise_dim):
    for epoch in range(epochs):
        start = time.time()
        for images in dataset:
            noise = tf.random_normal([BATCH_SIZE, noise_dim])
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)
                
                real_output = discriminator(images, training=True)
                generated_output = discriminator(generated_images, training=True)
                
                gen_loss = generator_loss(generated_output)
                dis_loss = discriminator_loss(real_output, generated_output)
            gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
            gradient_of_discriminator = disc_tape.gradient(dis_loss, discriminator.variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
            discriminator_optimizer.apply_gradients(zip(gradient_of_discriminator, discriminator.variables))
        if epoch % 1 == 0:
            display.clear_output(wait=True)
            generate_and_save_images(generator, epoch+1, random_vector_for_generation)
        if (epoch+1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        print("Time taken for epoch {} is {} sec".format(epoch+1, time.time()-start))
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, random_vector_for_generation)

if __name__ == "__main__":
    train(train_dataset, EPOCHS, noise_dim)
