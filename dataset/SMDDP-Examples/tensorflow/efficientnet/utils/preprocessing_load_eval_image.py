def load_eval_image(filename: Text, image_size: int=IMAGE_SIZE) ->tf.Tensor:
    """Reads an image from the filesystem and applies image preprocessing.

  Args:
    filename: a filename path of an image.
    image_size: image height/width dimension.

  Returns:
    A preprocessed and normalized image `Tensor`.
  """
    image_bytes = tf.io.read_file(filename)
    image = preprocess_for_eval(image_bytes, image_size)
    return image
