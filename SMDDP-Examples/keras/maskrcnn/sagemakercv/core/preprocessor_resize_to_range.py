def resize_to_range(image, masks=None, min_dimension=None, max_dimension=
    None, method=tf.image.ResizeMethod.BILINEAR, align_corners=False,
    pad_to_max_dimension=False):
    """Resizes an image so its dimensions are within the provided value.

  The output size can be described by two cases:
  1. If the image can be rescaled so its minimum dimension is equal to the
     provided value without the other dimension exceeding max_dimension,
     then do so.
  2. Otherwise, resize so the largest dimension is equal to max_dimension.

  Args:
    image: A 3D tensor of shape [height, width, channels]
    masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width] containing instance masks.
    min_dimension: (optional) (scalar) desired size of the smaller image
                   dimension.
    max_dimension: (optional) (scalar) maximum allowed size
                   of the larger image dimension.
    method: (optional) interpolation method used in resizing. Defaults to
            BILINEAR.
    align_corners: bool. If true, exactly align all 4 corners of the input
                   and output. Defaults to False.
    pad_to_max_dimension: Whether to resize the image and pad it with zeros
      so the resulting image is of the spatial size
      [max_dimension, max_dimension]. If masks are included they are padded
      similarly.

  Returns:
    Note that the position of the resized_image_shape changes based on whether
    masks are present.
    resized_image: A 3D tensor of shape [new_height, new_width, channels],
      where the image has been resized (with bilinear interpolation) so that
      min(new_height, new_width) == min_dimension or
      max(new_height, new_width) == max_dimension.
    resized_masks: If masks is not None, also outputs masks. A 3D tensor of
      shape [num_instances, new_height, new_width].
    resized_image_shape: A 1D tensor of shape [3] containing shape of the
      resized image.

  Raises:
    ValueError: if the image is not a 3D tensor.
  """
    if len(image.get_shape()) != 3:
        raise ValueError('Image should be 3D tensor')
    if image.get_shape().is_fully_defined():
        new_size = _compute_new_static_size(image, min_dimension, max_dimension
            )
    else:
        new_size = _compute_new_dynamic_size(image, min_dimension,
            max_dimension)
    new_image = tf.image.resize(image, new_size[:-1], method=method)
    if pad_to_max_dimension:
        new_image = tf.image.pad_to_bounding_box(new_image, 0, 0,
            max_dimension, max_dimension)
    result = [new_image]
    if masks is not None:
        new_masks = tf.expand_dims(masks, 3)
        new_masks = tf.image.resize(new_masks, new_size[:-1], method=tf.
            image.ResizeMethod.NEAREST_NEIGHBOR)
        new_masks = tf.squeeze(new_masks, 3)
        if pad_to_max_dimension:
            new_masks = tf.image.pad_to_bounding_box(new_masks, 0, 0,
                max_dimension, max_dimension)
        result.append(new_masks)
    result.append(new_size)
    return result
