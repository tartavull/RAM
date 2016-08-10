"""Recurrent Visual Attention Model. V. Mnih et al."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf

from glimpse import GlimpseNet, LocNet
from utils import weight_variable, bias_variable, loglikelihood
from config import Config

from tensorflow.examples.tutorials.mnist import input_data

logging.getLogger().setLevel(logging.INFO)

rnn_cell = tf.nn.rnn_cell
seq2seq = tf.nn.seq2seq

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

config = Config()
n_steps = config.step


loc_mean_arr = []
sampled_loc_arr = []


def get_next_input(output, i):
  loc, loc_mean = loc_net(output)
  gl_next = gl(loc)
  loc_mean_arr.append(loc_mean)
  sampled_loc_arr.append(loc)
  return gl_next


images_ph = tf.placeholder(
    tf.float32, (config.batch_size, config.original_size * config.original_size * config.num_channels))
labels_ph = tf.placeholder(tf.int64, (config.batch_size))

# Build the aux nets.
with tf.variable_scope('glimpse_net'):
  gl = GlimpseNet(config, images_ph)
with tf.variable_scope('loc_net'):
  loc_net = LocNet(config)

init_loc = tf.random_uniform((config.batch_size, 2), minval=-1, maxval=1)
init_glimpse = gl(init_loc)
# Core network.
lstm_cell = rnn_cell.LSTMCell(
    config.cell_size, config.g_size, num_proj=config.cell_out_size)
init_state = lstm_cell.zero_state(config.batch_size, tf.float32)
inputs = [init_glimpse]
inputs.extend([0] * (config.num_glimpses))
outputs, _ = seq2seq.rnn_decoder(
    inputs, init_state, lstm_cell, loop_function=get_next_input)
# Take the last step only.
output = outputs[-1]
# Build classification network.
with tf.variable_scope('cls'):
  w_logit = weight_variable((config.cell_output_size, config.num_classes))
  b_logit = bias_variable((config.num_classes,))
logits = tf.nn.xw_plus_b(output, w_logit, b_logit)

# cross-entropy.
xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels_ph)
xent = tf.reduce_mean(xent)

pred_labels = tf.argmax(logits, 1)
# 0/1 reward.
reward = tf.cast(tf.equal(pred_labels, labels_ph), tf.float32)
rewards = tf.expand_dims(reward, 1)  # [batch_sz, 1]
rewards = tf.tile(rewards, (1, config.num_glimpses))  # [batch_sz, timesteps]
logll = loglikelihood(loc_mean_arr, sampled_loc_arr, config.loc_std)
logllratio = tf.reduce_mean(logll * rewards)
reward = tf.reduce_mean(reward)

var_list = tf.trainable_variables()
# hybrid loss
loss = -logllratio + xent   # `-` for minimize
grads = tf.gradients(loss, var_list)
grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)
opt = tf.train.AdamOptimizer()
train_op = opt.apply_gradients(zip(grads, var_list))

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  for i in xrange(n_steps):
    images, labels = mnist.train.next_batch(config.batch_size)
    logllratio_val, reward_val, loss_val, _ = sess.run(
        [logllratio, reward, loss, train_op],
        feed_dict={
            images_ph: images, labels_ph: labels
        }
    )
    if i and i % 20 == 0:
      logging.info(
          'reward = {:3.4f}\tloss={:3.4f}'.format(reward_val, loss_val)
      )
