def pad_or_clip_nd(tensor, output_shape):
    """Pad or Clip given tensor to the output shape.

  Args:
    tensor: Input tensor to pad or clip.
    output_shape: A list of integers / scalar tensors (or None for dynamic dim)
      representing the size to pad or clip each dimension of the input tensor.

  Returns:
    Input tensor padded and clipped to the output shape.
  """
    tensor_shape = tf.shape(input=tensor)
    clip_size = [(tf.where(tensor_shape[i] - shape > 0, shape, -1) if shape
         is not None else -1) for i, shape in enumerate(output_shape)]
    clipped_tensor = tf.slice(tensor, begin=tf.zeros(len(clip_size), dtype=
        tf.int32), size=clip_size)
    clipped_tensor_shape = tf.shape(input=clipped_tensor)
    trailing_paddings = [(shape - clipped_tensor_shape[i] if shape is not
        None else 0) for i, shape in enumerate(output_shape)]
    paddings = tf.stack([tf.zeros(len(trailing_paddings), dtype=tf.int32),
        trailing_paddings], axis=1)
    padded_tensor = tf.pad(tensor=clipped_tensor, paddings=paddings)
    output_static_shape = [(dim if not isinstance(dim, tf.Tensor) else None
        ) for dim in output_shape]
    padded_tensor.set_shape(output_static_shape)
    return padded_tensor
