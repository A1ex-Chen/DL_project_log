def normalize_image(image):
    """Normalize the image.

    Args:
    image: a tensor of shape [height, width, 3] in dtype=tf.float32.

    Returns:
    normalized_image: a tensor which has the same shape and dtype as image,
      with pixel values normalized.
    """
    offset = tf.constant([0.485, 0.456, 0.406])
    offset = tf.reshape(offset, shape=(1, 1, 3))
    scale = tf.constant([0.229, 0.224, 0.225])
    scale = tf.reshape(scale, shape=(1, 1, 3))
    normalized_image = (image - offset) / scale
    return normalized_image
