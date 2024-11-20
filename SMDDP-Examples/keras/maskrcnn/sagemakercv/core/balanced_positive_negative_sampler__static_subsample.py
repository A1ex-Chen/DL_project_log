def _static_subsample(self, indicator, batch_size, labels):
    """Returns subsampled minibatch.

    Args:
      indicator: boolean tensor of shape [N] whose True entries can be sampled.
        N should be a complie time constant.
      batch_size: desired batch size. This scalar cannot be None.
      labels: boolean tensor of shape [N] denoting positive(=True) and negative
        (=False) examples. N should be a complie time constant.

    Returns:
      sampled_idx_indicator: boolean tensor of shape [N], True for entries which
        are sampled. It ensures the length of output of the subsample is always
        batch_size, even when number of examples set to True in indicator is
        less than batch_size.

    Raises:
      ValueError: if labels and indicator are not 1D boolean tensors.
    """
    if not indicator.shape.is_fully_defined():
        raise ValueError(
            'indicator must be static in shape when is_static isTrue')
    if not labels.shape.is_fully_defined():
        raise ValueError('labels must be static in shape when is_static isTrue'
            )
    if not isinstance(batch_size, int):
        raise ValueError(
            'batch_size has to be an integer when is_static isTrue.')
    input_length = tf.shape(input=indicator)[0]
    num_true_sampled = tf.reduce_sum(input_tensor=tf.cast(indicator, tf.
        float32))
    additional_false_sample = tf.less_equal(tf.cumsum(tf.cast(tf.
        logical_not(indicator), tf.float32)), batch_size - num_true_sampled)
    indicator = tf.logical_or(indicator, additional_false_sample)
    permutation = tf.random.shuffle(tf.range(input_length))
    indicator = ops.matmul_gather_on_zeroth_axis(tf.cast(indicator, tf.
        float32), permutation)
    labels = ops.matmul_gather_on_zeroth_axis(tf.cast(labels, tf.float32),
        permutation)
    indicator_idx = tf.where(tf.cast(indicator, tf.bool), tf.range(1, 
        input_length + 1), tf.zeros(input_length, tf.int32))
    signed_label = tf.where(tf.cast(labels, tf.bool), tf.ones(input_length,
        tf.int32), tf.scalar_mul(-1, tf.ones(input_length, tf.int32)))
    signed_indicator_idx = tf.multiply(indicator_idx, signed_label)
    sorted_signed_indicator_idx = tf.nn.top_k(signed_indicator_idx,
        input_length, sorted=True).values
    [num_positive_samples, num_negative_samples
        ] = self._get_num_pos_neg_samples(sorted_signed_indicator_idx,
        batch_size)
    sampled_idx = self._get_values_from_start_and_end(
        sorted_signed_indicator_idx, num_positive_samples,
        num_negative_samples, batch_size)
    sampled_idx = tf.abs(sampled_idx) - tf.ones(batch_size, tf.int32)
    sampled_idx = tf.multiply(tf.cast(tf.greater_equal(sampled_idx, tf.
        constant(0)), tf.int32), sampled_idx)
    sampled_idx_indicator = tf.cast(tf.reduce_sum(input_tensor=tf.one_hot(
        sampled_idx, depth=input_length), axis=0), tf.bool)
    reprojections = tf.one_hot(permutation, depth=input_length, dtype=tf.
        float32)
    return tf.cast(tf.tensordot(tf.cast(sampled_idx_indicator, tf.float32),
        reprojections, axes=[0, 0]), tf.bool)
