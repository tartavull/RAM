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
global_step = tf.get_variable(
    'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

# learning rate
training_steps_per_epoch = mnist.train.num_examples // config.batch_size
cn = CoreNet(config, global_step, training_steps_per_epoch)



all_summaries = []
all_summaries.extend([tf.scalar_summary(tag,var) for tag,var in [
  ('learning_rate', cn.learning_rate),
  ('baselines_mse',cn.baselines_mse),
  ('xent',cn.xent),
  ('logllratio',cn.logllratio),
  ('reward',cn.summary_rewards),
  ('loss',cn.loss),
  ('glimpse_error', cn.summary_glimpse_error)
  ]])

# print (cn.gl.extractions._shape_as_list)
# all_summaries.append(
#   tf.image_summary('glimpes', tf.concat(0,cn.gl.extractions),max_images=6))  
summary_op = tf.merge_summary(all_summaries)


#Just for tensorboard logging
squares_ph = tf.placeholder(tf.float32,
                            [1,config.original_size+2,
                             (config.original_size+1)*(config.num_glimpses+1)+1,3])


glimpses_ph = tf.placeholder(tf.float32,
                            [1,config.win_size+2,
                            (config.num_glimpses+1)*(config.win_size+1)+1,3])

outputs_ph = tf.placeholder(tf.float32,
                            [1,config.win_size+2,
                            (config.num_glimpses+1)*(config.win_size+1)+1,3])

glimpse_op = tf.merge_summary([tf.image_summary('squares',squares_ph, max_images=1),
                               tf.image_summary('glimpses',glimpses_ph, max_images=1),
                               tf.image_summary('output',outputs_ph, max_images=1)])

with tf.Session() as sess:
  summary_writer = tf.train.SummaryWriter(
               './log', graph=sess.graph)
  sess.run(tf.initialize_all_variables())
  for i in tqdm(xrange(config.step)):
    images, labels = mnist.train.next_batch(config.batch_size)
    cn.loc_net.samping = True
    sess.run(cn.train_op,
            feed_dict={
                cn.images_ph: images,
                cn.labels_ph: labels
            })

    if i and i % 50 == 0:
      loc, extractions, outputs, summary = sess.run(
        [tf.slice(cn.gl.summary_extraction_locs,
                  [0,0,0],
                  [config.num_glimpses+1,1,2]),
         tf.slice(cn.gl.extractions,
                  [0,0,0,0,0],
                  [config.num_glimpses+1,1,config.win_size,config.win_size,1]),
         tf.squeeze(tf.slice(cn.summary_image_output,
                  [0,0,0,0,0],
                  [config.num_glimpses+1,1,config.win_size,config.win_size,1])),
         summary_op],
              feed_dict={
                  cn.images_ph: images,
                  cn.labels_ph: labels
              })
      summary_writer.add_summary(summary, i)


      squares, glimpses, outputs = create_gimple_summary(loc, extractions, outputs, images, config)
      glimpse_summary = sess.run(glimpse_op,
        feed_dict={
          squares_ph: squares,
          glimpses_ph: glimpses,
          outputs_ph: outputs}
        )
      summary_writer.add_summary(glimpse_summary, i)
      summary_writer.flush()