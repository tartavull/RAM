import tensorflow as tf
from glimpse import GlimpseNet, LocNet

rnn_cell = tf.nn.rnn_cell
seq2seq = tf.nn.seq2seq


class Config(object):
  win_size = 8
  bandwidth = win_size ** 2
  batch_size = 32

config = Config()


def get_next_input(output, i):
  loc = loc_net(output)
  loc_next = gl(loc)
  return loc_next

init_loc = tf.random_uniform((config.batch_size, 2), minval=-1, maxval=1)
gl = GlimpseNet(config)
loc_net = LocNet(config)
init_glimpse = gl(init_loc)
lstm_cell = rnn_cell.LSTMCell(
  config.cell_size, config.g_size, num_proj=config.cell_out_size)
init_state = lstm_cell.zero_state(config.batch_size, tf.float32)
inputs = [init_glimpse]
inputs.extend([0] * (config.num_glimpses - 1))
outputs, _ = seq2seq.rnn_decoder(
  inputs, init_state, lstm_cell, loop_function=get_next_input)
