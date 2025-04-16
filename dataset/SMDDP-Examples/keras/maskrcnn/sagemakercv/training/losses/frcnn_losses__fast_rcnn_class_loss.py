def _fast_rcnn_class_loss(self, class_outputs, class_targets_one_hot,
    normalizer=1.0):
    """Computes classification loss."""
    with tf.name_scope('fast_rcnn_class_loss'):
        if self.num_classes == 1:
            class_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
                multi_class_labels=class_targets_one_hot, logits=
                class_outputs, label_smoothing=0.0, reduction=tf.compat.v1.
                losses.Reduction.MEAN)
        else:
            class_loss = _softmax_cross_entropy(onehot_labels=
                class_targets_one_hot, logits=class_outputs)
        if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
            class_loss /= normalizer
    return class_loss
