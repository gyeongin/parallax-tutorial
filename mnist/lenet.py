from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

learning_rate = 1e-3
weight_decay = 1e-4
num_classes = 10
dropout_keep_prob = 0.5

class LeNet(object):

  def __init__(self):
    images_ph = tf.placeholder(tf.float32, shape=[None, 784])
    labels_ph = tf.placeholder(tf.int64, shape=[None, num_classes])
    is_training_ph = tf.placeholder(tf.bool, shape=())
    self.images = images_ph
    self.labels = labels_ph
    self.is_training = is_training_ph

    global_step = tf.train.get_or_create_global_step()
    self.global_step = global_step

    images = tf.reshape(images_ph, [-1, 28, 28, 1])

    net = tf.layers.conv2d(images, 10, [5, 5],
                           activation=tf.nn.relu,
                           kernel_regularizer=tf.nn.l2_loss,
                           name='conv1')
    net = tf.layers.max_pooling2d(net, [2, 2], 2, name='pool1')
    net = tf.layers.conv2d(net, 20, [5, 5],
                           activation=tf.nn.relu,
                           kernel_regularizer=tf.nn.l2_loss,
                           name='conv2')
    net = tf.layers.max_pooling2d(net, [2, 2], 2, name='pool2')
    net = tf.layers.flatten(net)

    net = tf.layers.dense(net, 50,
                          activation=tf.nn.relu,
                          kernel_regularizer=tf.nn.l2_loss,
                          name='fc3')
    net = tf.layers.dropout(net, 1 - dropout_keep_prob, training=is_training_ph, name='dropout3')
    logits = tf.layers.dense(net, num_classes,
                             activation=None,
                             kernel_regularizer=tf.nn.l2_loss,
                             name='fc4')
    self.logits = logits

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_ph, logits=logits))
    loss += weight_decay * tf.losses.get_regularization_loss()
    self.loss = loss
    self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels_ph, axis=1)), tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    self.train_op = optimizer.minimize(loss, global_step=global_step)
