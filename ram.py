"""Recurrent Models of Visual Attention V. Mnih et al.

A recurrent neural network model that is capable
of extracting information from an image by adaptively selecting
a sequence of regions or locations and only processing the selected regions at
high resolution.

While the model is non-differentiable, it can be trained using reinforcement learning methods to
learn task-specific policies.


One important property of human perception is that one does not tend to process a whole scene
in its entirety at once. Instead humans focus attention selectively on parts of the visual space to
acquire information when and where it is needed, and combine information from different fixations
over time to build up an internal representation of the scene, guiding future eye movements
and decision making.

Maybe a reason for fixations is that are retinas are not convolutional, meaning that we cannot apply
the same feature map to the whole image. Also the resolution of the retina is only high in the fovea.

This model uses backpropagation to train the neural-network components and policy
gradient to address the non-differentiabilities due to the control problem of chosing the next region.

The agent can also
affect the true state of the environment by executing actions. Since the environment is only partially
observed the agent needs to integrate information over time in order to determine how to act and
how to deploy its sensor most effectively. At each step, the agent receives a scalar reward (which
depends on the actions the agent has executed and can be delayed), and the goal of the agent is to
maximize the total sum of such rewards.


The agent is built around a recurrent neural network. At each time step, it
processes the sensor data, integrates information over time, and chooses how to act and how to
deploy its sensor at next time step:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from glimpse import GlimpseNet, LocNet, CoreNet
from config import Config
from scipy.misc import imsave
from display_glimpses import create_gimple_summary

from tensorflow.examples.tutorials.mnist import input_data

logging.getLogger().setLevel(logging.INFO)


mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

config = Config()
n_steps = config.step
cn = CoreNet(config, mnist)

# learning rate
global_step = tf.get_variable(
    'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
training_steps_per_epoch = mnist.train.num_examples // config.batch_size
starter_learning_rate = config.lr_start
# decay per training epoch
learning_rate = tf.train.exponential_decay(
    starter_learning_rate,
    global_step,
    training_steps_per_epoch,
    0.97,
    staircase=True)
learning_rate = tf.maximum(learning_rate, config.lr_min)
opt = tf.train.AdamOptimizer(learning_rate)
train_op = opt.apply_gradients(zip(cn.grads, cn.var_list), global_step=global_step)


all_summaries = []
all_summaries.extend([tf.scalar_summary(tag,var) for tag,var in [
  ('learning_rate', learning_rate),
  ('baselines_mse',cn.baselines_mse),
  ('xent',cn.xent),
  ('logllratio',cn.logllratio),
  ('reward',cn.reward),
  ('loss',cn.loss),
  ]])

# print (cn.gl.extractions._shape_as_list)
# all_summaries.append(
#   tf.image_summary('glimpes', tf.concat(0,cn.gl.extractions),max_images=6))  
summary_op = tf.merge_summary(all_summaries)


squares_ph = tf.placeholder(tf.float32,
                            [1,config.original_size+2,
                             config.original_size*config.num_glimpses+config.num_glimpses+1,3])


glimpses_ph = tf.placeholder(tf.float32,
                            [1,config.win_size+2,
                            config.num_glimpses*(config.win_size+1)+1,3])

glimpse_op = tf.merge_summary([tf.image_summary('gls',squares_ph, max_images=1),
                               tf.image_summary('gls2',glimpses_ph, max_images=1)])



with tf.Session() as sess:
  summary_writer = tf.train.SummaryWriter(
               './log', graph=sess.graph)
  sess.run(tf.initialize_all_variables())
  for i in tqdm(xrange(n_steps)):
    images, labels = mnist.train.next_batch(config.batch_size)
    # duplicate M times, see Eqn (2)
    images = np.tile(images, [config.M, 1])
    labels = np.tile(labels, [config.M])
    cn.loc_net.samping = True
    sess.run(train_op,
            feed_dict={
                cn.images_ph: images,
                cn.labels_ph: labels
            })
    if i and i % 10 == 0:
      loc, extractions, summary = sess.run(
        [tf.slice(cn.gl.extraction_locs,[0,0,0],[config.num_glimpses,1,2]),
         tf.slice(cn.gl.extractions,[0,0,0,0,0],[config.num_glimpses,1,config.win_size,config.win_size,1]),
         summary_op],
              feed_dict={
                  cn.images_ph: images,
                  cn.labels_ph: labels
              })
      summary_writer.add_summary(summary, i)


      squares, glimpses = create_gimple_summary(loc, extractions, images, config)
      glimpse_summary = sess.run(glimpse_op,
        feed_dict={
          squares_ph: squares,
          glimpses_ph: glimpses}
        )
      summary_writer.add_summary(glimpse_summary, i)
      summary_writer.flush()
              
    if i and i % training_steps_per_epoch == 0:
      # Evaluation
      for dataset in [mnist.validation, mnist.test]:
        steps_per_epoch = dataset.num_examples // config.eval_batch_size
        correct_cnt = 0
        num_samples = steps_per_epoch * config.batch_size
        cn.loc_net.sampling = True
        for test_step in xrange(steps_per_epoch):
          images, labels = dataset.next_batch(config.batch_size)
          labels_bak = labels
          # Duplicate M times
          images = np.tile(images, [config.M, 1])
          labels = np.tile(labels, [config.M])
          softmax_val = sess.run(cn.softmax,
                                 feed_dict={
                                     cn.images_ph: images,
                                     cn.labels_ph: labels
                                 })
          softmax_val = np.reshape(softmax_val,
                                   [config.M, -1, config.num_classes])
          softmax_val = np.mean(softmax_val, 0)
          pred_labels_val = np.argmax(softmax_val, 1)
          pred_labels_val = pred_labels_val.flatten()
          correct_cnt += np.sum(pred_labels_val == labels_bak)
        acc = correct_cnt / num_samples
        if dataset == mnist.validation:
          logging.info('valid accuracy = {}'.format(acc))
        else:
          logging.info('test accuracy = {}'.format(acc))
