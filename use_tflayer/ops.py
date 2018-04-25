import tensorflow as tf
import numpy as np
import cv2
import os



def load_384_256_img(data_dir, path):
    img = cv2.imread(os.path.join(data_dir, path))
    img = cv2.resize(img, (256, 384), cv2.INTER_CUBIC)
    img = np.reshape(img, (1, -1))
    # img = np.reshape(mean_sub(img), (1, -1))
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

def make_log_dir(self, log_dir, pid):
        dir = os.path.join(log_dir, pid)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        os.mkdir(dir)
        return dir

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
