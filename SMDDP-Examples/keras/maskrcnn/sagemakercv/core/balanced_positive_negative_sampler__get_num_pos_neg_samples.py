def _get_num_pos_neg_samples(self, sorted_indices_tensor, sample_size):
    """Counts the number of positives and negatives numbers to be sampled.

    Args:
      sorted_indices_tensor: A sorted int32 tensor of shape [N] which contains
        the signed indices of the examples where the sign is based on the label
        value. The examples that cannot be sampled are set to 0. It samples
        atmost sample_size*positive_fraction positive examples and remaining
        from negative examples.
      sample_size: Size of subsamples.

    Returns:
      A tuple containing the number of positive and negative labels in the
      subsample.
    """
    input_length = tf.shape(input=sorted_indices_tensor)[0]
    valid_positive_index = tf.greater(sorted_indices_tensor, tf.zeros(
        input_length, tf.int32))
    num_sampled_pos = tf.reduce_sum(input_tensor=tf.cast(
        valid_positive_index, tf.int32))
    max_num_positive_samples = tf.constant(int(sample_size * self.
        _positive_fraction), tf.int32)
    num_positive_samples = tf.minimum(max_num_positive_samples, num_sampled_pos
        )
    num_negative_samples = tf.constant(sample_size, tf.int32
        ) - num_positive_samples
    return num_positive_samples, num_negative_samples
