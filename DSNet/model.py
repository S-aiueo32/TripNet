import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import glob
import os
import datetime
import time

from ops import *


class TripNet(object):
    def __init__(self, sess,
                 epoch=25,
                 batch_size=5,
                 data_list="./dataset.csv",
                 data_dir="../images/resize_google_triplet",
                 log_dir="./logs",
                 ckpt_dir="./ckpt",
                 train=False,
                 visualize=False,
                 visualize_dir="projector"):
        self.sess = sess

        self.epoch = epoch
        self.batch_size = batch_size
        self.data_list = data_list
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.visualize_dir = visualize_dir

        self.now = datetime.datetime.today().strftime("%Y%m%d%H%M%S")

        # Shared Network Parameters
        with tf.variable_scope('share'):
            with tf.variable_scope('deep'):
                with tf.variable_scope('conv1'):
                    with tf.variable_scope('cv1'):
                        self.W_d_conv1_1 = weight_variable(
                            [3, 3, 3, 64], name="weight")
                        self.b_d_conv1_1 = bias_variable([64], name="bias")
                    with tf.variable_scope('cv2'):
                        self.W_d_conv1_2 = weight_variable(
                            [3, 3, 64, 64], name="weight")
                        self.b_d_conv1_2 = bias_variable([64], name="bias")
                with tf.variable_scope('conv2'):
                    with tf.variable_scope('cv1'):
                        self.W_d_conv2_1 = weight_variable(
                            [3, 3, 64, 128], name="weight")
                        self.b_d_conv2_1 = bias_variable([128], name="bias")
                    with tf.variable_scope('cv2'):
                        self.W_d_conv2_2 = weight_variable(
                            [3, 3, 128, 128], name="weight")
                        self.b_d_conv2_2 = bias_variable([128], name="bias")
                with tf.variable_scope('conv3'):
                    with tf.variable_scope('cv1'):
                        self.W_d_conv3_1 = weight_variable(
                            [3, 3, 128, 256], name="weight")
                        self.b_d_conv3_1 = bias_variable([256], name="bias")
                    with tf.variable_scope('cv2'):
                        self.W_d_conv3_2 = weight_variable(
                            [3, 3, 256, 256], name="weight")
                        self.b_d_conv3_2 = bias_variable([256], name="bias")
                with tf.variable_scope('conv4'):
                    with tf.variable_scope('cv1'):
                        self.W_d_conv4_1 = weight_variable(
                            [3, 3, 256, 128], name="weight")
                        self.b_d_conv4_1 = bias_variable([128], name="bias")
                with tf.variable_scope('fc'):
                    self.W_d_fc = weight_variable(
                        [6 * 4 * 128, 128], name="weight")
                    self.b_d_fc = bias_variable([128], name="bias")
            with tf.variable_scope('shallow'):
                with tf.variable_scope('conv1'):
                    self.W_s_conv1 = weight_variable(
                        [8, 8, 3, 32], name="weight")
                    self.b_s_conv1 = bias_variable([32], name="bias")
                with tf.variable_scope('conv2'):
                    self.W_s_conv2 = weight_variable(
                        [8, 8, 3, 32], name="weight")
                    self.b_s_conv2 = bias_variable([32], name="bias")
                with tf.variable_scope('fc'):
                    self.W_s_fc = weight_variable(
                        [12 * 8 * 32 + 3 * 2 * 32, 96], name="weight")
                    self.b_s_fc = bias_variable([96], name="bias")
            with tf.variable_scope('assemble'):
                self.W_a = weight_variable([128 + 96, 128], name="weight")
                self.b_a = bias_variable([128], name="bias")

        self.x_q = tf.placeholder(tf.float32, [None, 294912])
        self.f_q = self.build_model(self.x_q, train=train, scope="query")

        self.vars_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        if train:
            self.x_p = tf.placeholder(tf.float32, [None, 294912])
            self.f_p = self.build_model(self.x_p, scope="positive")
            self.x_n = tf.placeholder(tf.float32, [None, 294912])
            self.f_n = self.build_model(self.x_n, scope="negative")
            self.loss, self.d_p, self.d_n = self.triplet_loss(
                self.f_q, self.f_p, self.f_n)
            self.summary = tf.summary.merge_all()

        self.saver = tf.train.Saver(self.vars_save, max_to_keep=None)
        self.sess.run(tf.global_variables_initializer())

    def build_model(self, x, train=True, scope=None):
        xr = tf.reshape(x, [-1, 384, 256, 3])
        with tf.variable_scope(scope):
            with tf.variable_scope('deep'):
                h_d_cv1_1 = relu(
                    conv2d(xr, self.W_d_conv1_1) + self.b_d_conv1_1)
                h_d_cv1_2 = relu(
                    conv2d(h_d_cv1_1, self.W_d_conv1_2) + self.b_d_conv1_2)
                h_d_cv1_drop = tf.nn.dropout(h_d_cv1_2, 0.75)
                h_d_cv2_pool = max_pool_4x4(h_d_cv1_drop)
                h_d_cv2_bn = batch_norm(h_d_cv2_pool, name='bn2', phase=True)
                h_d_cv2_1 = relu(
                    conv2d(h_d_cv2_bn, self.W_d_conv2_1) + self.b_d_conv2_1)
                h_d_cv2_2 = relu(
                    conv2d(h_d_cv2_1, self.W_d_conv2_2) + self.b_d_conv2_2)
                h_d_cv2_drop = tf.nn.dropout(h_d_cv2_2, 0.75)
                h_d_cv3_pool = max_pool_4x4(h_d_cv2_drop)
                h_d_cv3_bn = batch_norm(h_d_cv3_pool, name="bn3", phase=True)
                h_d_cv3_1 = relu(
                    conv2d(h_d_cv3_bn, self.W_d_conv3_1) + self.b_d_conv3_1)
                h_d_cv3_2 = relu(
                    conv2d(h_d_cv3_1, self.W_d_conv3_2) + self.b_d_conv3_2)
                h_d_cv3_drop = tf.nn.dropout(h_d_cv3_2, 0.75)
                h_d_cv4_pool = max_pool_4x4(h_d_cv3_drop)
                h_d_cv4_bn = batch_norm(h_d_cv4_pool, name="bn4", phase=True)
                h_d_cv4_1 = relu(
                    conv2d(h_d_cv4_bn, self.W_d_conv4_1) + self.b_d_conv4_1)
                h_d_cv4_f = tf.reshape(h_d_cv4_1, [-1, 6 * 4 * 128])
                h_d_fc = tf.matmul(h_d_cv4_f, self.W_d_fc) + self.b_d_fc
                h_d_fc_norm = tf.nn.l2_normalize(h_d_fc, 1)  # deep-output
            with tf.variable_scope('shallow'):
                h_s_sub1 = avg_pool_4x4(xr)
                h_s_cv1 = relu(conv2d(h_s_sub1, self.W_s_conv1,
                                      strides=[1, 4, 4, 1]) + self.b_s_conv1)
                h_s_cv1_pool = max_pool_3x3(h_s_cv1)
                h_s_cv1_flat = tf.reshape(h_s_cv1_pool, [-1, 12 * 8 * 32])
                h_s_sub2 = avg_pool_8x8(xr)
                h_s_cv2 = relu(conv2d(h_s_sub2, self.W_s_conv2,
                                      strides=[1, 4, 4, 1]) + self.b_s_conv2)
                h_s_cv2_pool = max_pool_7x7(h_s_cv2)
                h_s_cv2_flat = tf.reshape(h_s_cv2_pool, [-1, 3 * 2 * 32])
                h_s = tf.concat([h_s_cv1_flat, h_s_cv2_flat], 1)
                h_s_fc = tf.matmul(h_s, self.W_s_fc) + self.b_s_fc
                h_s_fc_norm = tf.nn.l2_normalize(h_s_fc, 1)  # shallow output
            with tf.variable_scope('assemble'):
                h_a = tf.concat([h_d_fc_norm, h_s_fc_norm], 1)
                h_a_fc = relu(tf.matmul(h_a, self.W_a) + self.b_a)
                h_a_norm = tf.nn.l2_normalize(h_a_fc, 1)  # shallow output
        return h_a_norm

    def train(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer().minimize(self.loss)

        writer = tf.summary.FileWriter(os.path.join(
            self.log_dir, self.now), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        dataset = np.loadtxt(self.data_list, delimiter=",", dtype='unicode')
        data_num = len(dataset)
        batch_num = int(np.ceil(data_num / self.batch_size))

        step = 0
        for epoch in range(self.epoch):
            np.random.shuffle(dataset)
            for batch in range(batch_num):
                t = time.time()
                start = self.batch_size * batch
                end = min(start + self.batch_size, data_num)
                img_q, img_p, img_n = get_batch(self.data_dir,
                                                dataset[start:end])

                _, train_summary = self.sess.run([train_step, self.summary], feed_dict={
                    self.x_q: img_q, self.x_p: img_p, self.x_n: img_n})
                writer.add_summary(train_summary, step)

                loss, d_p, d_n = self.sess.run([self.loss, self.d_p, self.d_n], feed_dict={
                                               self.x_q: img_q, self.x_p: img_p, self.x_n: img_n})
                t = time.time() - t
                print("[epoch:%d/batch:%d/total:%d] t_loss: %f, d_p: %f, d_n:%f, time:%f" %
                      (epoch, batch, step, np.mean(loss), np.mean(d_p), np.mean(d_n), t))
                step += 1
            self.save(self.ckpt_dir, step)

    def triplet_loss(self, f_q, f_p, f_n, margin=0.3):
        # compute loss
        d_p = tf.norm(f_q - f_p, axis=1)
        d_n = tf.norm(f_q - f_n, axis=1)
        loss = tf.reduce_mean(tf.maximum(margin + d_p - d_n, 0.0))
        #norm_p = tf.exp(tf.norm(f_q - f_p, axis=1))
        #norm_n = tf.exp(tf.norm(f_q - f_n, axis=1))
        #d_p = norm_p / (norm_p + norm_n)
        #d_n = norm_n / (norm_p + norm_n)
        #loss = 0.5 * (tf.square(d_p) + tf.square(1 - d_n))
        # logging
        d_p = tf.reduce_mean(d_p)
        d_n = tf.reduce_mean(d_n)
        tf.summary.scalar("D_p", d_p)
        tf.summary.scalar("D_n", d_n)
        tf.summary.scalar("Loss", loss)
        return loss, d_p, d_n

    def save(self, ckpt_dir, step):
        model_name = "TripNet.model"
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        if not os.path.exists(os.path.join(ckpt_dir, self.now)):
            os.makedirs(os.path.join(ckpt_dir, self.now))
        print("[!] Saving the model of step %d" % step)
        self.saver.save(self.sess, os.path.join(
            ckpt_dir, self.now, model_name), global_step=step)

    def load(self, load_step=None):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if load_step:
            model_name = "TripNet.model-%d" % load_step
            self.saver.restore(self.sess, os.path.join(
                self.ckpt_dir, model_name))
        elif ckpt:
            last_model = ckpt.model_checkpoint_path
            self.saver.restore(self.sess, last_model)
        return ckpt

    def visualize(self, width=64, height=96):
        if not os.path.exists(self.visualize_dir):
            os.makedirs(self.visualize_dir)

        files = glob.glob(self.data_dir + "/*.jpg")

        # encode
        print("[!] Vectorizing images...")
        feat_table = []
        meta_table = []
        for path in files:
            img = load_384_256_img_(path)
            feat = self.sess.run(self.f_q, feed_dict={self.x_q: img})
            feat_table.append([flatten for inner in feat for flatten in inner])
            meta_table.append(path)
        pd.DataFrame(feat_table).to_csv(os.path.join(self.visualize_dir, "vector.tsv"),
                                        sep="\t", index=False, header=False)
        pd.DataFrame(meta_table).to_csv(os.path.join(self.visualize_dir, "metadata.tsv"),
                                        sep="\t", index=False, header=False)

        # generate sprite image
        print("[!] Generating a sprite image...")
        generate_sprite(files, width=width, height=height)

        # projection
        print("[!] Running Embedding Projector...")
        from tensorflow.contrib.tensorboard.plugins import projector
        with tf.Session() as sess:
            vec = np.genfromtxt(os.path.join(
                self.visualize_dir, "./vector.tsv"))
            images = tf.Variable(vec, name='images')
            saver = tf.train.Saver([images])

            sess.run(images.initializer)
            saver.save(sess, os.path.join(self.visualize_dir, 'images.ckpt'))

            summary_writer = tf.summary.FileWriter(self.visualize_dir)

            config = projector.ProjectorConfig()

            embedding = config.embeddings.add()
            embedding.tensor_name = images.name
            if tf.__version__ == "1.1.0":
                embedding.metadata_path = os.path.join(
                    self.visualize_dir, "metadata.tsv")
                embedding.sprite.image_path = os.path.join(
                    self.visualize_dir, "sprite_img.jpg")
            else:
                embedding.metadata_path = 'metadata.tsv'
                embedding.sprite.image_path = "sprite_img.jpg"
            embedding.sprite.single_image_dim.extend([width, height])

            projector.visualize_embeddings(summary_writer, config)

        print("[!] Processes have just finished!")
        print("[NEXT] Please run TensorBoard with the argument \"--logdir ./projector\"")
