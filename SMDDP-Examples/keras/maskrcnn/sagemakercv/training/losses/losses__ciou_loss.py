def _ciou_loss(y_true, y_pred, weights):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, y_pred.dtype)
    weights = tf.cast(weights, tf.float32)
    y_true = tf.reshape(y_true, [-1, 4])
    y_pred = tf.reshape(y_pred, [-1, 4])
    weights = tf.reshape(weights, [-1, 4])
    ciou = _calculate_ciou(b1=y_true, b2=y_pred, mode='ciou')
    ciou_loss = 1.0 - ciou
    ciou_loss = tf.tile(tf.expand_dims(ciou_loss, -1), [1, 4]) * weights
    ciou_loss = tf.math.divide_no_nan(tf.math.reduce_sum(ciou_loss), tf.
        math.count_nonzero(weights, dtype=tf.float32))
    assert ciou_loss.dtype == tf.float32
    return ciou_loss
