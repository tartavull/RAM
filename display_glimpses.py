import numpy as np

def create_sequence_of_images_with_borders(height, width, n_seq):
    total_height = height+2
    total_width = (width+1)*n_seq + 1
    img = np.zeros(shape=(total_height, total_width,3),dtype=np.uint8)

    #Add stripes to delimit the individual images
    img[0,:,2] = 255
    img[-1,:,2] = 255
    img[:,::width+1,2] = 255
    return img

def glimpse_location_mapping(loc, height, win_size):
    """
    Asumes that the original image is square
    """

    #the center of the original image is position 0, -1 is leftmost 
    #and 1 the rightmost pixel 
    pixel_coordinates = (loc + 1.0) * height/2
    top_left_corner = pixel_coordinates - np.array([win_size/2, win_size/2])
    top_left_corner = np.minimum(top_left_corner, height)
    top_left_corner = np.maximum(top_left_corner, 0)
    return top_left_corner.astype(np.uint8)

def add_background_img(n_seq ,seq_img, background):

    for i in xrange(n_seq):
        background_width = background.shape[0]
        x_offset = (background_width + 1) * i + 1

        #rgb channels to add as grayscale
        seq_img[1:-1, x_offset:x_offset+background_width, 0] = background
        seq_img[1:-1, x_offset:x_offset+background_width, 1] = background
        seq_img[1:-1, x_offset:x_offset+background_width, 2] = background

    return seq_img

def get_square_corners(top_left, height, win_size):
    """
    We want an square which contains the glimpse
    """
    def clamp(value):
        return min(height, max(0,value))

    top_left = clamp(top_left[1]-1) , clamp(top_left[0]-1)
    top_right = clamp(top_left[0] + win_size + 2), top_left[1]
    bottom_left = top_left[0] , clamp(top_left[1] + win_size + 2)
    bottom_right = top_right[0], bottom_left[1]

    return top_left, top_right, bottom_right, bottom_left

def draw_square_in_image(img, original_size, seq, top_left, top_right, bottom_right, bottom_left):
    x_offset= 1 + (original_size + 1) * seq
    img[top_left[1], top_left[0]+x_offset:top_right[0]+x_offset,1] = 255
    img[top_left[1]:bottom_left[1], top_left[0]+x_offset,1] = 255
    img[bottom_left[1], bottom_left[0]+x_offset:bottom_right[0]+x_offset,1] = 255
    img[top_right[1]:bottom_right[1], bottom_right[0]+x_offset,1] = 255
    return img

def create_gimple_summary(loc, extractions, images, config):

    squares = create_sequence_of_images_with_borders(config.original_size,
                                                 config.original_size,
                                                 config.num_glimpses)

    background = images[0].reshape(config.original_size, config.original_size)*255.0
    background = background.astype(np.uint8)
    squares = add_background_img(config.num_glimpses, squares, background)
    loc = glimpse_location_mapping(loc, config.original_size, config.win_size)

    for i in xrange(config.num_glimpses):
        top_left, top_right, bottom_right, bottom_left = get_square_corners(
            loc[i,0,:], config.original_size, config.win_size)
        squares = draw_square_in_image(
                squares, config.original_size, i,
                top_left, top_right, bottom_right, bottom_left)

    squares = np.expand_dims(squares,0)

    glimpses = create_sequence_of_images_with_borders(config.win_size,
                                                      config.win_size,
                                                      config.num_glimpses)
    for i in xrange(config.num_glimpses):
        x_offset = 1 + i * (config.win_size+1)
        image = extractions[i] * 255.0
        image = image.astype(np.uint8)
        glimpses[1:-1,x_offset:config.win_size+x_offset,0] = image[0,:,:,0]
        glimpses[1:-1,x_offset:config.win_size+x_offset,1] = image[0,:,:,0]
        glimpses[1:-1,x_offset:config.win_size+x_offset,2] = image[0,:,:,0]
    glimpses = np.expand_dims(glimpses,0)

    return squares , glimpses