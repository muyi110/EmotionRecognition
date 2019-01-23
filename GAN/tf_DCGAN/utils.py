# -*- coding:UTF-8 -*-
import math
import random
import scipy.misc
import numpy as np
import pprint
from time import gmtime, strftime
import tensorflow.contrib.slim as slim
import tensorflow as tf

pp = pprint.PrettyPrinter()

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width, resize_height=64, 
              resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale=False):
    if grayscale:
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image) / 127.5 -1

def inverse_transform(images):
    return (images+1.)/2.

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3,4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
          i = idx % size[1]
          j = idx // size[1]
          img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
          i = idx % size[1]
          j = idx // size[1]
          img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def image_manifold_size(num_images):
     manifold_h = int(np.floor(np.sqrt(num_images)))
     manifold_w = int(np.ceil(np.sqrt(num_images)))
     assert manifold_h * manifold_w == num_images
     return manifold_h, manifold_w

def visualize(sess, dcgan, config):
    image_frame_dim = int(math.ceil(config.batch_size**.5))
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in range(dcgan.z_dim):
        print(" [*] %d" % idx)
        z_sample = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))
        for kdx, z in enumerate(z_sample):
            z[idx] = values[kdx]

        if config.dataset == "mnist":
            y = np.random.choice(10, config.batch_size)
            y_one_hot = np.zeros((config.batch_size, 10))
            y_one_hot[np.arange(config.batch_size), y] = 1
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
        else:
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))
