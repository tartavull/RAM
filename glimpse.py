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
    self.win_size = config.win_size
    self.bandwidth = config.bandwidth

    self.images_ph = images_ph
    self.init_weights(config)
    self.extractions = []
    self.summary_extraction_locs = []
  def init_weights(self, config):
    """ Initialize all the trainable weights."""
    self.w_g0 = weight_variable((config.bandwidth, config.input_rnn))
    self.b_g0 = bias_variable((config.input_rnn,))

    self.w_l0 = weight_variable((config.loc_dim, config.input_rnn))
    self.b_l0 = bias_variable((config.input_rnn,))

    self.w_g1 = weight_variable((config.input_rnn, config.hidden_rnn))
    self.b_g1 = bias_variable((config.hidden_rnn,))

    self.w_l1 = weight_variable((config.input_rnn, config.hidden_rnn))
    self.b_l1 = weight_variable((config.hidden_rnn,))

  def get_glimpse(self, loc):
    """Take glimpse on the original images."""
    imgs = tf.reshape(self.images_ph, [
        tf.shape(self.images_ph)[0], self.original_size, self.original_size,
        self.num_channels
    ])

    extracted = tf.image.extract_glimpse(
      imgs,[self.win_size, self.win_size], loc)
   
    glimpse_imgs = tf.reshape(extracted, [
        tf.shape(loc)[0], self.bandwidth
    ])

    # just for logging to tensorboard.
    self.extractions.append(extracted)
    self.summary_extraction_locs.append(loc)
    return glimpse_imgs

  def __call__(self, loc):
    """Given a location, the glimpse is extraced.
    And some processing is that before inputting to the
    recurrent neural network.
    
    Args:
        loc (tensor): location of the glimpse
    
    Returns:
        tensor: input to the hidden layer of the recurrent network
    """
    glimpse_input = self.get_glimpse(loc)
    glimpse_input = tf.reshape(glimpse_input,
                               (tf.shape(loc)[0], self.bandwidth))
    g = tf.nn.relu(tf.nn.xw_plus_b(glimpse_input, self.w_g0, self.b_g0))
    g = tf.nn.xw_plus_b(g, self.w_g1, self.b_g1)
    l = tf.nn.relu(tf.nn.xw_plus_b(loc, self.w_l0, self.b_l0))
    l = tf.nn.xw_plus_b(l, self.w_l1, self.b_l1)
    g = tf.nn.relu(g + l)
    return g


class LocNet(object):
  """Location network.

  Take output from other network and produce and sample the next location.
  The location network is always trained with REINFORCE.

  As you can see in call, the gradient is stopped here.
  """
  def __init__(self, config):
    self.loc_dim = config.loc_dim
    self.loc_std = config.loc_std
    self._sampling = True

    with tf.name_scope('LocNet'):
      self.w = weight_variable((config.output_rnn,  config.loc_dim))
      self.b = bias_variable((config.loc_dim,))

  def __call__(self, input):
    """
    The location in and x,y pair between -1 and 1.
    Reinforcement learning algorithm needs to choose between exploration and
    exploitation.
    Sampling adds noise to the input location to make the network explore other options.
    Args:
        input (tensor): hidden layer of the recurrent network
    
    Returns:
        (tensor, tensor): Location of next glimpse, desired location of next glimpse
        Both are the same in the canse of no explorartion (sampling is false).
    """
    with tf.name_scope('LocNet'):
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

  def __init__(self, config, global_step, training_steps_per_epoch):
    self.training_steps_per_epoch = training_steps_per_epoch
    self.global_step = global_step
    self.batch_size = config.batch_size
    self.hidden_rnn = config.hidden_rnn
    self.output_rnn = config.output_rnn
    self.num_glimpses = config.num_glimpses
    self.bandwidth = config.bandwidth
    self.win_size = config.win_size

    # useful for summarizing
    self.loc_mean_arr = []
    self.sampled_loc_arr = []
    self.next_inputs = []
    
    self.create_placeholders(config)
    self.create_auxiliary_networks(config)
    init_glimpse = self.get_first_glimpse()
    recurrent_outputs = self.create_recurrent_network(init_glimpse)
    baselines =  self.create_baselines(recurrent_outputs)
    glimpse_error = self.create_image_output(recurrent_outputs)
   

    with tf.name_scope('loss'):
      # if we reduce_sum the reward will be proportional to number of pixels correct
      # making it follow big trunks
      self.xent = tf.reduce_mean(glimpse_error)
      rewards = 1 - glimpse_error

      # where does the weight of the location network learns?
      logll = loglikelihood(self.loc_mean_arr, self.sampled_loc_arr, config.loc_std)
      self.advs = rewards - tf.stop_gradient(baselines)
      self.logllratio = tf.reduce_mean(logll * self.advs)
      
      self.summary_rewards = tf.reduce_mean(rewards)

      # We want the baseline to get closer to actual reward 
      self.baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))

      var_list = tf.trainable_variables()
      # hybrid loss
      self.loss = -self.logllratio + self.xent + self.baselines_mse  # `-` for minimize
      grads = tf.gradients(self.loss, var_list)
      grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)

      # decay per training epoch
      learning_rate = tf.train.exponential_decay(
          config.lr_start,
          self.global_step,
          self.training_steps_per_epoch,
          0.97,
          staircase=True)
      self.learning_rate = tf.maximum(learning_rate, config.lr_min)

      opt = tf.train.AdamOptimizer(self.learning_rate)
      self.train_op = opt.apply_gradients(zip(grads, var_list), global_step=self.global_step)


  def get_first_glimpse(self):
    # 0.0 is the center of the image while -1 and 1 are the extrems
    # when taking glimpses
    init_loc = tf.random_uniform((self.batch_size, 2), minval=-1, maxval=1)
    return self.gl(init_loc)

  def create_recurrent_network(self, init_glimpse):
    # Core network.
    # Note that it sees num_glimpses + 1, but we are ignoring the first one
    # because it was not randomly chosen. (TODO verify)

    with tf.name_scope('recurrent_network'):
      lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_rnn, state_is_tuple=True)
      init_state = lstm_cell.zero_state(self.batch_size, tf.float32)
      inputs = [init_glimpse]
      inputs.extend([0] * (self.num_glimpses))
      outputs, _ = tf.nn.seq2seq.rnn_decoder(
        inputs, init_state, lstm_cell, loop_function=self.get_next_input)
      return outputs # [timesteps, batch_sz]

  def create_baselines(self, recurrent_outputs):
    # Time independent baselines
    # we want to reward only if larger than the expected reward
    with tf.variable_scope('baseline'):
      w_baseline = weight_variable((self.output_rnn, 1))
      b_baseline = bias_variable((1,))
      baselines = []
      for t, output in enumerate(recurrent_outputs[1:]):
        baseline_t = tf.nn.xw_plus_b(output, w_baseline, b_baseline)
        baseline_t = tf.squeeze(baseline_t)
        baselines.append(baseline_t)
      baselines = tf.pack(baselines)  # [timesteps, batch_sz]
      baselines = tf.transpose(baselines)  # [batch_size, timesteps]
      return baselines

  def create_image_output(self, recurrent_outputs):
    """
    After the network looks at a glimpse, it outputs a binary image containing
    the object currently being traced. A reward can be generated at each glimpse
    by comping how similar the output image is to the ground truth for that given
    object.
    
    Returns:
        TYPE: Description
    """
    with tf.name_scope('image_output'):
      w_logit = weight_variable((self.output_rnn, self.bandwidth))
      b_logit = bias_variable((self.bandwidth,))

      #reshape to perform the matrix multiplication
      reshaped_outputs = tf.reshape(recurrent_outputs,[self.batch_size * (self.num_glimpses+1),self.output_rnn])
      logits_image = tf.nn.xw_plus_b(reshaped_outputs, w_logit, b_logit)
      

      # create tensor just for visualization
      sigmoid = tf.nn.sigmoid(logits_image)
      self.summary_image_output = tf.reshape(sigmoid,[self.num_glimpses+1,
                                              self.batch_size,
                                              self.win_size,
                                              self.win_size, 1])
      
      # create an error for each glimpse
      desired_output = tf.reshape(self.gl.extractions,[self.batch_size, 
                                                       self.num_glimpses+1,
                                                       self.bandwidth])

      reshaped_sigmoid = tf.reshape(sigmoid, [self.batch_size, 
                                              self.num_glimpses+1,
                                              self.bandwidth])

      glimpse_error = tf.reduce_mean(
          tf.square(reshaped_sigmoid-desired_output))


      self.summary_glimpse_error = glimpse_error

      return glimpse_error

  def get_next_input(self, output, i):
    loc, loc_mean = self.loc_net(output)
    gl_next = self.gl(loc)
    self.next_inputs.append(gl_next)
    self.loc_mean_arr.append(loc_mean)
    self.sampled_loc_arr.append(loc)
    return gl_next

  def create_placeholders(self, config):
    self.batch_size = config.batch_size
    self.images_ph = tf.placeholder(tf.float32,
                               [config.batch_size, config.original_size * config.original_size *
                                config.num_channels])
    
    self.labels_ph = tf.placeholder(tf.int64, [config.batch_size])


  def create_auxiliary_networks(self, config):
    with tf.name_scope('glimpse_net'):
      self.gl = GlimpseNet(config, self.images_ph)
      self.loc_net = LocNet(config)
