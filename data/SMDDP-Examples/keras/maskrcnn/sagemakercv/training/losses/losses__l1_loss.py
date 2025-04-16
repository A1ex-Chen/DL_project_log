def _l1_loss(y_true, y_pred, weights, delta=0.0):
    l1_loss = tf.compat.v1.losses.absolute_difference(y_true, y_pred,
        weights=weights)
    assert l1_loss.dtype == tf.float32
    DEBUG_LOSS_IMPLEMENTATION = False
    if DEBUG_LOSS_IMPLEMENTATION:
        mlperf_loss = tf.compat.v1.losses.huber_loss(y_true, y_pred,
            weights=weights, delta=delta, reduction=tf.compat.v1.losses.
            Reduction.SUM_BY_NONZERO_WEIGHTS)
        print_op = tf.print('Huber Loss - MLPerf:', mlperf_loss,
            ' && Legacy Loss:', l1_loss)
        with tf.control_dependencies([print_op]):
            l1_loss = tf.identity(l1_loss)
    return l1_loss
