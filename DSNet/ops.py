import tensorflow as tf
import numpy as np
import cv2
import os
import glob


def conv2d(x, W, strides=[1, 1, 1, 1], name=None):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME', name=name)


def max_pool_3x3(x, name=None):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool_4x4(x, name=None):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                          strides=[1, 4, 4, 1], padding='SAME', name=name)


def max_pool_7x7(x, name=None):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 7, 7, 1],
                          strides=[1, 4, 4, 1], padding='SAME', name=name)


def avg_pool_4x4(x, name=None):
    return tf.nn.avg_pool(x, ksize=[1, 4, 4, 1],
                          strides=[1, 4, 4, 1], padding='SAME', name=name)


def avg_pool_8x8(x, name=None):
    return tf.nn.avg_pool(x, ksize=[1, 8, 8, 1],
                          strides=[1, 8, 8, 1], padding='SAME', name=name)


def weight_variable(shape, name=None):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.2)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def batch_norm(x, name, reuse=False, phase=True):
    return tf.contrib.layers.batch_norm(x,
                                        # decay=0.9,
                                        # epsilon=1e-5,
                                        scale=True,
                                        updates_collections=None,
                                        is_training=phase,
                                        trainable=phase,
                                        reuse=reuse,
                                        scope=name)


def relu(x, name=None):
    return tf.nn.relu(x, name)


def l_relu(x, alpha=0.2, name=None):
    return tf.nn.leaky_relu(x, alpha, name)


def load_384_256_img(data_dir, path):
    img = cv2.imread(os.path.join(data_dir, path))
    img = cv2.resize(img, (256, 384), cv2.INTER_CUBIC)
    img = np.reshape(mean_sub(img), (1, -1))
    return img


def load_384_256_img_(path):
    img = cv2.imread(os.path.join(path))
    img = cv2.resize(img, (256, 384), cv2.INTER_CUBIC)
    img = np.reshape(mean_sub(img), (1, -1))
    return img


def mean_sub(img):
    for i in range(3):
        ch = img[:, :, i]
        img[:, :, i] = (ch - np.mean(ch)) / np.std(ch)
    return img


def get_batch(data_dir, dataset):
    img_q = img_p = img_n = np.empty((0, 294912))
    for path in dataset:
        img_q = np.append(img_q, load_384_256_img(data_dir, path[0]), axis=0)
        img_p = np.append(img_p, load_384_256_img(data_dir, path[1]), axis=0)
        img_n = np.append(img_n, load_384_256_img(data_dir, path[2]), axis=0)
    return img_q, img_p, img_n


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


def remove_log(log_dir):
    for path in glob.glob(os.path.join(log_dir, '*')):
        os.remove(path)
