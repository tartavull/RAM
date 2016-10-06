from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import weight_variable, bias_variable, loglikelihood


class GlimpseNet(object):
  """Glimpse network.

  Take glimpse location input and output features for RNN.
  
  Given the location `loc` and input image `image_ph`, 
  uses the glimpse sensor to extract retina representation.

  The retina representation and glimpse location is then mapped into a hidden
  space using independent linear layers parameterized by g0,g1 and l0,l1 respectively
  using rectified units followed by another linear layer to combine the information 
  from both components.

  The external input to the recurrent neural network is the glimpse feature vector
  At each step, the agent performs two actions: it decides how to deploy its sensor via the
  sensor control lt, and an environment action at which might affect the state of the environment.
  """

  def __init__(self, config, images_ph):
    self.original_size = config.original_size
    self.num_channels = config.num_channels
    self.sensor_size = config.sensor_size
    self.win_size = config.win_size

    self.images_ph = images_ph
    self.init_weights(config)
    self.extractions = []
    self.extraction_locs = []
  def init_weights(self, config):
    """ Initialize all the trainable weights."""
    self.w_g0 = weight_variable((config.sensor_size, config.hg_size))
    self.b_g0 = bias_variable((config.hg_size,))

    self.w_l0 = weight_variable((config.loc_dim, config.hl_size))
    self.b_l0 = bias_variable((config.hl_size,))

    self.w_g1 = weight_variable((config.hg_size, config.g_size))
    self.b_g1 = bias_variable((config.g_size,))

    self.w_l1 = weight_variable((config.hl_size, config.g_size))
    self.b_l1 = weight_variable((config.g_size,))

  def get_glimpse(self, loc):
    """Take glimpse on the original images."""
    imgs = tf.reshape(self.images_ph, [
        tf.shape(self.images_ph)[0], self.original_size, self.original_size,
        self.num_channels
    ])

    extracted = tf.image.extract_glimpse(imgs,
                                         [self.win_size, self.win_size], loc)
    self.extractions.append(extracted)
    self.extraction_locs.append(loc)
    glimpse_imgs = tf.reshape(extracted, [
        tf.shape(loc)[0], self.win_size * self.win_size * self.num_channels
    ])
    return glimpse_imgs

  def __call__(self, loc):
    glimpse_input = self.get_glimpse(loc)
    glimpse_input = tf.reshape(glimpse_input,
                               (tf.shape(loc)[0], self.sensor_size))
    g = tf.nn.relu(tf.nn.xw_plus_b(glimpse_input, self.w_g0, self.b_g0))
    g = tf.nn.xw_plus_b(g, self.w_g1, self.b_g1)
    l = tf.nn.relu(tf.nn.xw_plus_b(loc, self.w_l0, self.b_l0))
    l = tf.nn.xw_plus_b(l, self.w_l1, self.b_l1)
    g = tf.nn.relu(g + l)
    return g


class LocNet(object):
  """Location network.

  Take output from other network and produce and sample the next location.

  """
  def __init__(self, config):
    self.loc_dim = config.loc_dim
    self.loc_std = config.loc_std
    self._sampling = True

    self.init_weights(config)

  def init_weights(self, config):
    self.w = weight_variable((config.cell_out_size,  config.loc_dim))
    self.b = bias_variable((config.loc_dim,))

  def __call__(self, input):
    mean = tf.clip_by_value(tf.nn.xw_plus_b(input, self.w, self.b), -1., 1.)
    mean = tf.stop_gradient(mean)
    if self._sampling:
      loc = mean + tf.random_normal(
          (tf.shape(input)[0], self.loc_dim), stddev=self.loc_std)
      loc = tf.clip_by_value(loc, -1., 1.)
    else:
      loc = mean
    loc = tf.stop_gradient(loc)
    return loc, mean

  @property
  def sampling(self):
    return self._sampling

  @sampling.setter
  def sampling(self, sampling):
    self._sampling = sampling



class CoreNet(object):

  def __init__(self, config, mnist):
    self.loc_mean_arr = []
    self.sampled_loc_arr = []
    self.next_inputs = []
    self.create_placeholders(config)
    self.create_auxiliary_networks(config)

    # 0.0 is the center of the image while -1 and 1 are the extrems
    # when taking glimpses
    init_loc = tf.random_uniform((self.n_examples, 2), minval=-1, maxval=1)
    init_glimpse = self.gl(init_loc)

    # Core network.
    lstm_cell = tf.nn.rnn_cell.LSTMCell(config.cell_size, state_is_tuple=True)
    init_state = lstm_cell.zero_state(self.n_examples, tf.float32)
    inputs = [init_glimpse]
    inputs.extend([0] * (config.num_glimpses))
    outputs, _ = tf.nn.seq2seq.rnn_decoder(
      inputs, init_state, lstm_cell, loop_function=self.get_next_input)


    # Time independent baselines
    # we want to reward only if larger than the expected reward
    with tf.variable_scope('baseline'):
      w_baseline = weight_variable((config.cell_out_size, 1))
      b_baseline = bias_variable((1,))
    baselines = []
    for t, output in enumerate(outputs[1:]):
      baseline_t = tf.nn.xw_plus_b(output, w_baseline, b_baseline)
      baseline_t = tf.squeeze(baseline_t)
      baselines.append(baseline_t)
    baselines = tf.pack(baselines)  # [timesteps, batch_sz]
    baselines = tf.transpose(baselines)  # [batch_sz, timesteps]

    # Take the last step only.
    output = outputs[-1]
    # Build classification network.
    with tf.variable_scope('cls'):
      w_logit = weight_variable((config.cell_out_size, config.num_classes))
      b_logit = bias_variable((config.num_classes,))
    logits = tf.nn.xw_plus_b(output, w_logit, b_logit)
    self.softmax = tf.nn.softmax(logits)

    # cross-entropy.
    self.xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.labels_ph)
    self.xent = tf.reduce_mean(self.xent)
    pred_labels = tf.argmax(logits, 1)
    # 0/1 reward.
    reward = tf.cast(tf.equal(pred_labels, self.labels_ph), tf.float32)
    rewards = tf.expand_dims(reward, 1)  # [batch_sz, 1]
    rewards = tf.tile(rewards, (1, config.num_glimpses))  # [batch_sz, timesteps]
    logll = loglikelihood(self.loc_mean_arr, self.sampled_loc_arr, config.loc_std)
    self.advs = rewards - tf.stop_gradient(baselines)
    self.logllratio = tf.reduce_mean(logll * self.advs)
    self.reward = tf.reduce_mean(reward)

    self.baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
    self.var_list = tf.trainable_variables()
    # hybrid loss
    self.loss = -self.logllratio + self.xent + self.baselines_mse  # `-` for minimize
    grads = tf.gradients(self.loss, self.var_list)
    self.grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)

 

  def get_next_input(self, output, i):
    loc, loc_mean = self.loc_net(output)
    gl_next = self.gl(loc)
    self.next_inputs.append(gl_next)
    self.loc_mean_arr.append(loc_mean)
    self.sampled_loc_arr.append(loc)
    return gl_next

  def create_placeholders(self, config):
    self.images_ph = tf.placeholder(tf.float32,
                               [None, config.original_size * config.original_size *
                                config.num_channels])
    self.n_examples = tf.shape(self.images_ph)[0] # number of examples
    
    self.labels_ph = tf.placeholder(tf.int64, [None])


  def create_auxiliary_networks(self, config):
    with tf.variable_scope('glimpse_net'):
      self.gl = GlimpseNet(config, self.images_ph)
    with tf.variable_scope('loc_net'):
      self.loc_net = LocNet(config)
