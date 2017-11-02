import numpy as np
import tensorflow as tf
from model import TripNet

tf.reset_default_graph()

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_integer("batch_size", 5, "The size of batch images [5]")
flags.DEFINE_string("data_list", "./dataset.csv",
                    "The data list (.csv file) [./dataset.csv]")
flags.DEFINE_string("data_dir", "../images/resize_google_triplet",
                    "Directory name that contains dataset [../images/resize_google_triplet]")
flags.DEFINE_string("log_dir", "./log",
                    "Directory name to save the logs [./log]")
flags.DEFINE_string("ckpt_dir", "./ckpt",
                    "Directory name to save the checkpoints [./ckpt]")
flags.DEFINE_boolean("train", False,
                     "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False,
                     "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("load_step", None, "The step you wanna load [None]")
FLAGS = flags.FLAGS


def main(_):
    with tf.Session() as sess:
        tripnet = TripNet(
            sess,
            epoch=FLAGS.epoch,
            batch_size=FLAGS.batch_size,
            data_list=FLAGS.data_list,
            data_dir=FLAGS.data_dir,
            ckpt_dir=FLAGS.ckpt_dir,
            log_dir=FLAGS.log_dir,
            train=FLAGS.train,
            visualize=FLAGS.visualize)

        if FLAGS.train:
            tripnet.train()
        if FLAGS.visualize:
            if not tf.train.get_checkpoint_state(FLAGS.ckpt_dir):
                raise Exception("[!] Train a model first, then run test mode")
            if not FLAGS.train:
                tripnet.load(FLAGS.load_step)
            tripnet.visualize()


if __name__ == "__main__":
    tf.app.run()
