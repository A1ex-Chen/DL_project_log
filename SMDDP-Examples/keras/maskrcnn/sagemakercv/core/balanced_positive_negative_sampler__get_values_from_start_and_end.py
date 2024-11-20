def _get_values_from_start_and_end(self, input_tensor, num_start_samples,
    num_end_samples, total_num_samples):
    """slices num_start_samples and last num_end_samples from input_tensor.

    Args:
      input_tensor: An int32 tensor of shape [N] to be sliced.
      num_start_samples: Number of examples to be sliced from the beginning
        of the input tensor.
      num_end_samples: Number of examples to be sliced from the end of the
        input tensor.
      total_num_samples: Sum of is num_start_samples and num_end_samples. This
        should be a scalar.

    Returns:
      A tensor containing the first num_start_samples and last num_end_samples
      from input_tensor.

    """
    input_length = tf.shape(input=input_tensor)[0]
    start_positions = tf.less(tf.range(input_length), num_start_samples)
    end_positions = tf.greater_equal(tf.range(input_length), input_length -
        num_end_samples)
    selected_positions = tf.logical_or(start_positions, end_positions)
    selected_positions = tf.cast(selected_positions, tf.float32)
    indexed_positions = tf.multiply(tf.cumsum(selected_positions),
        selected_positions)
    one_hot_selector = tf.one_hot(tf.cast(indexed_positions, tf.int32) - 1,
        total_num_samples, dtype=tf.float32)
    return tf.cast(tf.tensordot(tf.cast(input_tensor, tf.float32),
        one_hot_selector, axes=[0, 0]), tf.int32)
