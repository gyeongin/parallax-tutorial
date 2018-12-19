import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data
from lenet import LeNet

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of iterations to run for each workers.""")
tf.app.flags.DEFINE_integer('log_frequency', 50,
                            """How many steps between two logs.""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Batch size""")

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

model = LeNet()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  start = time.time()
  for i in range(FLAGS.max_steps):
    batch = mnist.train.next_batch(FLAGS.batch_size)
    _, loss = sess.run([model.train_op, model.loss], feed_dict={model.images: batch[0],
                                                                model.labels: batch[1],
                                                                model.is_training: True})
    if i % FLAGS.log_frequency == 0:
      end = time.time()
      throughput = float(FLAGS.log_frequency) / float(end - start)
      acc = sess.run(model.acc, feed_dict={model.images: mnist.test.images,
                                           model.labels: mnist.test.labels,
                                           model.is_training: False})
      print("step %d, test accuracy %lf, throughput: %f steps/sec" % (i, acc, throughput))
      start = time.time()
