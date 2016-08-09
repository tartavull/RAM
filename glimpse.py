from utils import weight_varialbes


class GlimpseNet(object):

  def __init__(self, config):
    self.hg_size = config.hg_size
    self.hl_size = config.hl_size
    self.g_size = config.g_size

    self.init_weights()

  def init_weights(self):
    pass

  def __call__(self, loc):
    pass
