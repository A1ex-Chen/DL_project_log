def _huber_loss(y_true, y_pred, weights, delta, reduction=ReductionV2.SUM):
    num_non_zeros = tf.math.count_nonzero(weights, dtype=tf.float32)
    huber_keras_loss = tf.keras.losses.Huber(delta=delta, reduction=
        reduction, name='huber_loss')
    if LooseVersion(tf.__version__) >= LooseVersion('2.2.0'):
        y_true = tf.expand_dims(y_true, axis=-1)
        y_pred = tf.expand_dims(y_pred, axis=-1)
    huber_loss = huber_keras_loss(y_true, y_pred, sample_weight=weights)
    assert huber_loss.dtype == tf.float32
    huber_loss = tf.math.divide_no_nan(huber_loss, num_non_zeros, name=
        'huber_loss')
    assert huber_loss.dtype == tf.float32
    if DEBUG_LOSS_IMPLEMENTATION:
        mlperf_loss = tf.compat.v1.losses.huber_loss(y_true, y_pred,
            weights=weights, delta=delta, reduction=tf.compat.v1.losses.
            Reduction.SUM_BY_NONZERO_WEIGHTS)
        print_op = tf.print('Huber Loss - MLPerf:', mlperf_loss,
            ' && Legacy Loss:', huber_loss)
        with tf.control_dependencies([print_op]):
            huber_loss = tf.identity(huber_loss)
    return huber_loss
