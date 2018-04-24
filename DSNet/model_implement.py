import tensorflow as tf

tf.reset_default_graph()


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


def avg_pool_16x16(x, name=None):
    return tf.nn.avg_pool(x, ksize=[1, 16, 16, 1],
                          strides=[1, 16, 16, 1], padding='SAME', name=name)


def avg_pool_32x32(x, name=None):
    return tf.nn.avg_pool(x, ksize=[1, 32, 32, 1],
                          strides=[1, 32, 32, 1], padding='SAME', name=name)


def weight_variable(shape, name=None):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.2)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def batch_norm(x, name=None, reuse=False, phase=True):
    return tf.contrib.layers.batch_norm(x,
                                        # decay=0.9,
                                        # epsilon=1e-5,
                                        scale=True,
                                        updates_collections=None,
                                        is_training=phase,
                                        trainable=phase,
                                        reuse=reuse,
                                        scope=name)


if __name__ == '__main__':
    with tf.variable_scope('share'):
        with tf.variable_scope('deep'):
            with tf.variable_scope('conv1'):
                with tf.variable_scope('cv1'):
                    W_d_conv1_1 = weight_variable([3, 3, 3, 64], name="weight")
                    b_d_conv1_1 = bias_variable([64], name="bias")
                with tf.variable_scope('cv2'):
                    W_d_conv1_2 = weight_variable(
                        [3, 3, 64, 64], name="weight")
                    b_d_conv1_2 = bias_variable([64], name="bias")
            with tf.variable_scope('conv2'):
                with tf.variable_scope('cv1'):
                    W_d_conv2_1 = weight_variable(
                        [3, 3, 64, 128], name="weight")
                    b_d_conv2_1 = bias_variable([128], name="bias")
                with tf.variable_scope('cv2'):
                    W_d_conv2_2 = weight_variable(
                        [3, 3, 128, 128], name="weight")
                    b_d_conv2_2 = bias_variable([128], name="bias")
            with tf.variable_scope('conv3'):
                with tf.variable_scope('cv1'):
                    W_d_conv3_1 = weight_variable(
                        [3, 3, 128, 256], name="weight")
                    b_d_conv3_1 = bias_variable([256], name="bias")
                with tf.variable_scope('cv2'):
                    W_d_conv3_2 = weight_variable(
                        [3, 3, 256, 256], name="weight")
                    b_d_conv3_2 = bias_variable([256], name="bias")
            with tf.variable_scope('conv4'):
                with tf.variable_scope('cv1'):
                    W_d_conv4_1 = weight_variable(
                        [3, 3, 256, 128], name="weight")
                    b_d_conv4_1 = bias_variable([128], name="bias")
            with tf.variable_scope('fc'):
                W_d_fc = weight_variable([6 * 4 * 128, 128], name="weight")
                b_d_fc = bias_variable([128], name="bias")
        with tf.variable_scope('shallow'):
            with tf.variable_scope('conv1'):
                W_s_conv1 = weight_variable([8, 8, 3, 32], name="weight")
                b_s_conv1 = bias_variable([32], name="bias")
            with tf.variable_scope('conv2'):
                W_s_conv2 = weight_variable([8, 8, 3, 32], name="weight")
                b_s_conv2 = bias_variable([32], name="bias")
            with tf.variable_scope('fc'):
                W_s_fc = weight_variable(
                    [12 * 8 * 32 + 3 * 2 * 32, 96], name="weight")
                b_s_fc = bias_variable([96], name="bias")
        with tf.variable_scope('assemble'):
            W_a = weight_variable([128 + 96, 128], name="weight")
            b_a = bias_variable([128], name="bias")

    x = tf.placeholder(tf.float32, [None, 294912])
    xr = tf.reshape(x, [-1, 384, 256, 3])

    with tf.variable_scope('net'):
        with tf.variable_scope('deep'):
            h_d_cv1_1 = conv2d(xr, W_d_conv1_1) + b_d_conv1_1
            h_d_cv1_2 = conv2d(h_d_cv1_1, W_d_conv1_2) + b_d_conv1_2
            h_d_cv1_drop = tf.nn.dropout(h_d_cv1_2, 0.75)
            h_d_cv2_pool = max_pool_4x4(h_d_cv1_drop)
            h_d_cv2_bn = batch_norm(h_d_cv2_pool, name='bn2', phase=True)
            h_d_cv2_1 = conv2d(h_d_cv2_bn, W_d_conv2_1) + b_d_conv2_1
            h_d_cv2_2 = conv2d(h_d_cv2_1, W_d_conv2_2) + b_d_conv2_2
            h_d_cv2_drop = tf.nn.dropout(h_d_cv2_2, 0.75)
            h_d_cv3_pool = max_pool_4x4(h_d_cv2_drop)
            h_d_cv3_bn = batch_norm(h_d_cv3_pool, name="bn3", phase=True)
            h_d_cv3_1 = conv2d(h_d_cv3_bn, W_d_conv3_1) + b_d_conv3_1
            h_d_cv3_2 = conv2d(h_d_cv3_1, W_d_conv3_2) + b_d_conv3_2
            h_d_cv3_drop = tf.nn.dropout(h_d_cv3_2, 0.75)
            h_d_cv4_pool = max_pool_4x4(h_d_cv3_drop)
            h_d_cv4_bn = batch_norm(h_d_cv4_pool, name="bn4", phase=True)
            h_d_cv4_1 = conv2d(h_d_cv4_bn, W_d_conv4_1) + b_d_conv4_1
            h_d_cv4_f = tf.reshape(h_d_cv4_1, [-1, 6 * 4 * 128])
            h_d_fc = tf.matmul(h_d_cv4_f, W_d_fc) + b_d_fc
            h_d_fc_norm = tf.nn.l2_normalize(h_d_fc, 1)  # deep-output
        with tf.variable_scope('shallow'):
            h_s_sub1 = avg_pool_4x4(xr)
            h_s_cv1 = conv2d(h_s_sub1, W_s_conv1, strides=[
                             1, 4, 4, 1]) + b_s_conv1
            h_s_cv1_pool = max_pool_3x3(h_s_cv1)
            h_s_cv1_flat = tf.reshape(h_s_cv1_pool, [-1, 12 * 8 * 32])

            h_s_sub2 = avg_pool_8x8(xr)
            h_s_cv2 = conv2d(h_s_sub2, W_s_conv2, strides=[
                             1, 4, 4, 1]) + b_s_conv2
            h_s_cv2_pool = max_pool_7x7(h_s_cv2)
            h_s_cv2_flat = tf.reshape(h_s_cv2_pool, [-1, 3 * 2 * 32])

            h_s = tf.concat([h_s_cv1_flat, h_s_cv2_flat], 1)
            h_s_fc = tf.matmul(h_s, W_s_fc) + b_s_fc
            h_s_norm = tf.nn.l2_normalize(h_s_fc, 1)  # shallow output
        with tf.variable_scope('assemble'):
            h_a = tf.concat([h_d_fc_norm, h_s_norm], 1)
            h_a_fc = tf.matmul(h_a, W_a) + b_a
            h_a_norm = tf.nn.l2_normalize(h_a_fc, 1)  # shallow output

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #writer = tf.summary.FileWriter('./impl', sess.graph)
