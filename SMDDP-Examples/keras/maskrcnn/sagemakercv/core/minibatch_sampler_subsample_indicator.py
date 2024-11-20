@staticmethod
def subsample_indicator(indicator, num_samples):
    """Subsample indicator vector.

    Given a boolean indicator vector with M elements set to `True`, the function
    assigns all but `num_samples` of these previously `True` elements to
    `False`. If `num_samples` is greater than M, the original indicator vector
    is returned.

    Args:
      indicator: a 1-dimensional boolean tensor indicating which elements
        are allowed to be sampled and which are not.
      num_samples: int32 scalar tensor

    Returns:
      a boolean tensor with the same shape as input (indicator) tensor
    """
    indices = tf.where(indicator)
    indices = tf.random.shuffle(indices)
    indices = tf.reshape(indices, [-1])
    num_samples = tf.minimum(tf.size(input=indices), num_samples)
    selected_indices = tf.slice(indices, [0], tf.reshape(num_samples, [1]))
    selected_indicator = ops.indices_to_dense_vector(selected_indices, tf.
        shape(input=indicator)[0])
    return tf.equal(selected_indicator, 1)
