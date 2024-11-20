def __call__(self, mask_outputs, mask_targets, select_class_targets):
    with tf.name_scope('mask_loss'):
        batch_size, num_masks, mask_height, mask_width = (mask_outputs.
            get_shape().as_list())
        weights = tf.tile(tf.reshape(tf.greater(select_class_targets, 0), [
            batch_size, num_masks, 1, 1]), [1, 1, mask_height, mask_width])
        weights = tf.cast(weights, tf.float32)
        loss = _sigmoid_cross_entropy(multi_class_labels=mask_targets,
            logits=mask_outputs, weights=weights, sum_by_non_zeros_weights=
            True, label_smoothing=self.label_smoothing)
        mrcnn_loss = self.mrcnn_weight_loss_mask * loss
    return mrcnn_loss
