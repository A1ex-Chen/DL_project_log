def _sigmoid_cross_entropy(multi_class_labels, logits, weights,
    sum_by_non_zeros_weights=False, label_smoothing=0.0):
    assert weights.dtype == tf.float32
    sigmoid_cross_entropy = tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=multi_class_labels, logits=logits,
        label_smoothing=label_smoothing, reduction=tf.compat.v1.losses.
        Reduction.NONE)
    assert sigmoid_cross_entropy.dtype == tf.float32
    sigmoid_cross_entropy = tf.math.multiply(sigmoid_cross_entropy, weights)
    sigmoid_cross_entropy = tf.math.reduce_sum(sigmoid_cross_entropy)
    assert sigmoid_cross_entropy.dtype == tf.float32
    if sum_by_non_zeros_weights:
        num_non_zeros = tf.math.count_nonzero(weights, dtype=tf.float32)
        sigmoid_cross_entropy = tf.math.divide_no_nan(sigmoid_cross_entropy,
            num_non_zeros, name='sum_by_non_zeros_weights')
    assert sigmoid_cross_entropy.dtype == tf.float32
    if DEBUG_LOSS_IMPLEMENTATION:
        if sum_by_non_zeros_weights:
            reduction = tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        else:
            reduction = tf.compat.v1.losses.Reduction.SUM
        mlperf_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
            multi_class_labels=multi_class_labels, logits=logits, weights=
            weights, reduction=reduction)
        print_op = tf.print('Sigmoid X-Entropy Loss (%s) - MLPerf:' %
            reduction, mlperf_loss, ' && Legacy Loss:', sigmoid_cross_entropy)
        with tf.control_dependencies([print_op]):
            sigmoid_cross_entropy = tf.identity(sigmoid_cross_entropy)
    return sigmoid_cross_entropy
