def _get_rpn_samples(self, match_results):
    """Computes anchor labels.

    This function performs subsampling for foreground (fg) and background (bg)
    anchors.
    Args:
      match_results: A integer tensor with shape [N] representing the
        matching results of anchors. (1) match_results[i]>=0,
        meaning that column i is matched with row match_results[i].
        (2) match_results[i]=-1, meaning that column i is not matched.
        (3) match_results[i]=-2, meaning that column i is ignored.
    Returns:
      score_targets: a integer tensor with the a shape of [N].
        (1) score_targets[i]=1, the anchor is a positive sample.
        (2) score_targets[i]=0, negative. (3) score_targets[i]=-1, the anchor is
        don't care (ignore).
    """
    sampler = (balanced_positive_negative_sampler.
        BalancedPositiveNegativeSampler(positive_fraction=self.
        _rpn_fg_fraction, is_static=False))
    indicator = tf.greater(match_results, -2)
    labels = tf.greater(match_results, -1)
    samples = sampler.subsample(indicator, self._rpn_batch_size_per_im, labels)
    positive_labels = tf.where(tf.logical_and(samples, labels), tf.constant
        (2, dtype=tf.int32, shape=match_results.shape), tf.constant(0,
        dtype=tf.int32, shape=match_results.shape))
    negative_labels = tf.where(tf.logical_and(samples, tf.logical_not(
        labels)), tf.constant(1, dtype=tf.int32, shape=match_results.shape),
        tf.constant(0, dtype=tf.int32, shape=match_results.shape))
    ignore_labels = tf.fill(match_results.shape, -1)
    return (ignore_labels + positive_labels + negative_labels,
        positive_labels, negative_labels)
