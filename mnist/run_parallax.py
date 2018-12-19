import os
import time
import tensorflow as tf
import parallax

from lenet import LeNet
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('resource_info_file',
                           os.path.abspath(os.path.join(
                               os.path.dirname(__file__),
                               '.',
                               'resource_info')),
                           'Filename containing cluster information')
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of iterations to run for each workers.""")
tf.app.flags.DEFINE_integer('log_frequency', 50,
                            """How many steps between two logs.""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Batch size""")
tf.app.flags.DEFINE_boolean('sync', True, '')

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Build single-GPU LeNet model
single_gpu_graph = tf.Graph()
with single_gpu_graph.as_default():
  model = LeNet()

parallax_config = parallax.Config()
ckpt_config = parallax.CheckPointConfig(ckpt_dir='ckpt',
                                        save_ckpt_steps=FLAGS.log_frequency)
parallax_config.ckpt_config = ckpt_config

sess, num_workers, worker_id, num_replicas_per_worker = parallax.parallel_run(
    single_gpu_graph,
    FLAGS.resource_info_file,
    sync=FLAGS.sync,
    parallax_config=parallax_config)

start = time.time()
for i in range(FLAGS.max_steps):
  batch = mnist.train.next_batch(FLAGS.batch_size, shuffle=False)
  _, loss = sess.run([model.train_op, model.loss], feed_dict={model.images: [batch[0]],
                                                              model.labels: [batch[1]],
                                                              model.is_training: [True]})
  if i % FLAGS.log_frequency == 0:
    end = time.time()
    throughput = float(FLAGS.log_frequency) / float(end - start)
    acc = sess.run(model.acc, feed_dict={model.images: [mnist.test.images],
                                         model.labels: [mnist.test.labels],
                                         model.is_training: [False]})[0]
    parallax.log.info("step: %d, test accuracy: %lf, throughput: %f steps/sec"
                      % (i, acc, throughput))
    start = time.time()
