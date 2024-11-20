def _flip_image(image):
    image_flipped = tf.image.flip_left_right(image)
    return image_flipped
