def _rpn_score_loss(self, score_outputs, score_targets, normalizer=1.0):
    """Computes score loss."""
    with tf.name_scope('rpn_score_loss'):
        mask = tf.math.greater_equal(score_targets, 0)
        mask = tf.cast(mask, dtype=tf.float32)
        score_targets = tf.maximum(score_targets, tf.zeros_like(score_targets))
        score_targets = tf.cast(score_targets, dtype=tf.float32)
        assert score_outputs.dtype == tf.float32
        assert score_targets.dtype == tf.float32
        score_loss = _sigmoid_cross_entropy(multi_class_labels=
            score_targets, logits=score_outputs, weights=mask,
            sum_by_non_zeros_weights=False, label_smoothing=self.
            label_smoothing)
        assert score_loss.dtype == tf.float32
        if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
            score_loss /= normalizer
        assert score_loss.dtype == tf.float32
    return score_loss
