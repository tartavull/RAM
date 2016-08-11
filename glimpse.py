from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import weight_variable, bias_variable


class GlimpseNet(object):
  """Glimpse network.

  Take glimpse location input and output features for RNN.

  """

  def __init__(self, config, images_ph):
    self.original_size = config.original_size
    self.num_channels = config.num_channels
    self.sensor_size = config.sensor_size
    self.win_size = config.win_size
    self.minRadius = config.minRadius
    self.depth = config.depth

    self.batch_size = config.batch_size

    self.hg_size = config.hg_size
    self.hl_size = config.hl_size
    self.g_size = config.g_size
    self.loc_dim = config.loc_dim

    self.images_ph = images_ph

    self.init_weights()

  def init_weights(self):
    """ Initialize all the trainable weights."""
    self.w_g0 = weight_variable((self.sensor_size, self.hg_size))
    self.b_g0 = bias_variable((self.hg_size,))
    self.w_l0 = weight_variable((self.loc_dim, self.hl_size))
    self.b_l0 = bias_variable((self.hl_size,))
    self.w_g1 = weight_variable((self.hg_size, self.g_size))
    self.b_g1 = bias_variable((self.g_size,))
    self.w_l1 = weight_variable((self.hl_size, self.g_size))
    self.b_l1 = weight_variable((self.g_size,))

  def get_glimpse(self, loc):
    """Take glimpse on the original images."""
    loc = tf.round(((loc + 1) / 2) * self.original_size)
    loc = tf.cast(loc, tf.int32)
    img = tf.reshape(self.images_ph, (self.batch_size, self.original_size,
                                      self.original_size, self.num_channels))
    zooms = []
    # process each image individually
    for k in xrange(self.batch_size):
      imgZooms = []
      one_img = img[k, :, :, :]
      max_radius = self.minRadius * (2**(self.depth - 1))
      offset = 2 * max_radius
      # pad image with zeros
      one_img = tf.image.pad_to_bounding_box(
          one_img, offset, offset, max_radius * 4 + self.original_size,
          max_radius * 4 + self.original_size)
      for i in xrange(self.depth):
        r = int(self.minRadius * (2**(i)))
        d_raw = 2 * r
        d = tf.constant(d_raw, shape=[1])
        d = tf.tile(d, [2])
        loc_k = loc[k, :]
        adjusted_loc = offset + loc_k - r
        one_img2 = tf.reshape(one_img, (one_img.get_shape()[0].value,
                                        one_img.get_shape()[1].value))
        # crop image to (d x d)
        zoom = tf.slice(one_img2, adjusted_loc, d)
        # resize cropped image to (sensorBandwidth x sensorBandwidth)
        zoom = tf.image.resize_bilinear(
            tf.reshape(zoom, (1, d_raw, d_raw, 1)),
            (self.win_size, self.win_size))
        zoom = tf.reshape(zoom, (self.win_size, self.win_size))
        imgZooms.append(zoom)
      zooms.append(tf.pack(imgZooms))
    zooms = tf.pack(zooms)
    return zooms

  def __call__(self, loc):
    glimpse_input = self.get_glimpse(loc)
    glimpse_input = tf.reshape(glimpse_input,
                               (self.batch_size, self.sensor_size))
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
    self.input_dim = config.cell_output_size
    self.loc_std = config.loc_std
    self.batch_size = config.batch_size
    self._sampling = True

    self.init_weights()

  def init_weights(self):
    self.w = weight_variable((self.input_dim, self.loc_dim))
    self.b = bias_variable((self.loc_dim,))

  def __call__(self, input):
    mean = tf.nn.tanh(tf.nn.xw_plus_b(input, self.w, self.b))
    if self._sampling:
      loc = mean + tf.random_normal(
          (self.batch_size, self.loc_dim), stddev=self.loc_std)
    else:
      loc = mean
    return loc, mean

  @property
  def sampling(self):
    return self._sampling

  @sampling.setter
  def sampling(self, sampling):
    self._sampling = sampling
