def subsample(self, indicator, batch_size, labels, scope=None):
    """Returns subsampled minibatch.

    Args:
      indicator: boolean tensor of shape [N] whose True entries can be sampled.
      batch_size: desired batch size. If None, keeps all positive samples and
        randomly selects negative samples so that the positive sample fraction
        matches self._positive_fraction. It cannot be None is is_static is True.
      labels: boolean tensor of shape [N] denoting positive(=True) and negative
          (=False) examples.
      scope: name scope.

    Returns:
      sampled_idx_indicator: boolean tensor of shape [N], True for entries which
        are sampled.

    Raises:
      ValueError: if labels and indicator are not 1D boolean tensors.
    """
    if len(indicator.get_shape().as_list()) != 1:
        raise ValueError(
            'indicator must be 1 dimensional, got a tensor of shape %s' %
            indicator.get_shape())
    if len(labels.get_shape().as_list()) != 1:
        raise ValueError(
            'labels must be 1 dimensional, got a tensor of shape %s' %
            labels.get_shape())
    if labels.dtype != tf.bool:
        raise ValueError('labels should be of type bool. Received: %s' %
            labels.dtype)
    if indicator.dtype != tf.bool:
        raise ValueError('indicator should be of type bool. Received: %s' %
            indicator.dtype)
    if self._is_static:
        return self._static_subsample(indicator, batch_size, labels)
    else:
        negative_idx = tf.logical_not(labels)
        positive_idx = tf.logical_and(labels, indicator)
        negative_idx = tf.logical_and(negative_idx, indicator)
        if batch_size is None:
            max_num_pos = tf.reduce_sum(input_tensor=tf.cast(positive_idx,
                dtype=tf.int32))
        else:
            max_num_pos = int(self._positive_fraction * batch_size)
        sampled_pos_idx = self.subsample_indicator(positive_idx, max_num_pos)
        num_sampled_pos = tf.reduce_sum(input_tensor=tf.cast(
            sampled_pos_idx, tf.int32))
        if batch_size is None:
            negative_positive_ratio = (1 - self._positive_fraction
                ) / self._positive_fraction
            max_num_neg = tf.cast(negative_positive_ratio * tf.cast(
                num_sampled_pos, dtype=tf.float32), dtype=tf.int32)
        else:
            max_num_neg = batch_size - num_sampled_pos
        sampled_neg_idx = self.subsample_indicator(negative_idx, max_num_neg)
        return tf.logical_or(sampled_pos_idx, sampled_neg_idx)
