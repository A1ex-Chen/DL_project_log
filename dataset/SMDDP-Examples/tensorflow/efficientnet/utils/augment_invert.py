def invert(image: tf.Tensor) ->tf.Tensor:
    """Inverts the image pixels."""
    image = tf.convert_to_tensor(image)
    return 255 - image
