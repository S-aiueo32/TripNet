import tensorflow as tf
import numpy as np
import cv2
import os

def conv2d(x, W, name=None):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)


def max_pool_4x4(x, name=None):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                          strides=[1, 4, 4, 1], padding='SAME', name=name)


def weight_variable(shape, name=None):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.2)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def batch_norm(x, name, reuse=False, is_training=True, trainable=True):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        epsilon=1e-5,
                                        scale=True,
                                        updates_collections=None,
                                        is_training=is_training,
                                        trainable=trainable,
                                        reuse=reuse,
                                        scope=name)


def generate_sprite(files, save_dir="projector", width=64, height=96, max_size=8192):
    h_num = np.int(np.ceil(np.sqrt(len(files))))
    sprite_img = np.zeros((height, width, 3), np.uint8)
    row_img = np.zeros((height, width, 3), np.uint8)
    for i, path in enumerate(files):
        img = cv2.resize(cv2.imread(path), (width, height))
        if i < h_num:
            if i % h_num == 0:
                sprite_img = img
            else:
                sprite_img = cv2.hconcat([sprite_img, img])
        else:
            if i % h_num == 0:
                row_img = img
            else:
                row_img = cv2.hconcat([row_img, img])
            if i + 1 == len(files):
                margin = (h_num - 1) - (i % h_num)
                for _ in range(margin):
                    blank = np.zeros((height, width, 3), np.uint8)
                    row_img = cv2.hconcat([row_img, blank])
                    i += 1
                if sprite_img.shape[0] != sprite_img.shape[1]:
                    blank = np.zeros(
                        (height, sprite_img.shape[1], 3), np.uint8)
                    row_img = cv2.vconcat([row_img, blank])
            if i % h_num == h_num - 1:
                sprite_img = cv2.vconcat([sprite_img, row_img])
    cv2.imwrite(os.path.join(save_dir, "sprite_img.jpg"), sprite_img)


def show_variables():
    print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
