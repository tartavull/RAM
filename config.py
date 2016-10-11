class Config(object):
  
  # Because we are working on MNIST images are grayscale, and of size 28x28
  num_channels = 1
  original_size = 28
  # MNIST has 10 classes
  num_classes = 10

  # Size of the window for the gimplse, in oposition to the paper, 
  # the glimpse take is not multiscale (retina-like).
  win_size = 28

  # We can think of the glimpse sensor as bandwitch limited
  bandwidth = win_size**2 * num_channels
  
  # Number of episodes to play at the same time
  # Given that we are doing policy gradient we need to average to 
  # get a sample of the correct gradients
  batch_size = 2
  num_glimpses = 5
  max_grad_norm = 5.
  loc_dim = 2
  
  loc_std = 0.22
  minRadius = 8


  # What is the difference of all this things? 
  input_rnn = 128
  hidden_rnn = 1024
  output_rnn = hidden_rnn

  # How many training iterations to have
  step = 100000

  # Paramenters for an exponentially decaying learning rate 
  lr_start = 1e-3
  lr_min = 1e-4
