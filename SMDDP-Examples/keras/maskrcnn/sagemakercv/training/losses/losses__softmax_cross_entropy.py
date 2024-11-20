def _softmax_cross_entropy(onehot_labels, logits, label_smoothing=0.0):
    num_non_zeros = tf.math.count_nonzero(onehot_labels, dtype=tf.float32)
    if label_smoothing == 0.0:
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels
            =onehot_labels, logits=logits)
    else:
        softmax_cross_entropy = tf.compat.v1.losses.softmax_cross_entropy(
            onehot_labels, logits, label_smoothing=label_smoothing,
            reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
    assert softmax_cross_entropy.dtype == tf.float32
    if label_smoothing == 0.0:
        softmax_cross_entropy = tf.math.reduce_sum(softmax_cross_entropy)
        softmax_cross_entropy = tf.math.divide_no_nan(softmax_cross_entropy,
            num_non_zeros, name='softmax_cross_entropy')
    assert softmax_cross_entropy.dtype == tf.float32
    DEBUG_LOSS_IMPLEMENTATION = False
    if DEBUG_LOSS_IMPLEMENTATION:
        mlperf_loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels
            =onehot_labels, logits=logits, reduction=tf.compat.v1.losses.
            Reduction.SUM_BY_NONZERO_WEIGHTS)
        print_op = tf.print('Softmax X-Entropy Loss - MLPerf:', mlperf_loss,
            ' && Legacy Loss:', softmax_cross_entropy)
        with tf.control_dependencies([print_op]):
            softmax_cross_entropy = tf.identity(softmax_cross_entropy)
    return softmax_cross_entropy
