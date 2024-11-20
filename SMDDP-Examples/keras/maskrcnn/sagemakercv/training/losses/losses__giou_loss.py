def _giou_loss(y_true, y_pred, weights, reduction='sum'):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, y_pred.dtype)
    weights = tf.cast(weights, tf.float32)
    y_true = tf.reshape(y_true, [-1, 4])
    y_pred = tf.reshape(y_pred, [-1, 4])
    weights = tf.reshape(weights, [-1, 4])
    giou = _calculate_giou(y_true, y_pred)
    giou_loss = 1.0 - giou
    giou_loss = tf.tile(tf.expand_dims(giou_loss, -1), [1, 4]) * weights
    avg_factor = tf.math.count_nonzero(weights, dtype=tf.float32)
    if reduction == 'sum':
        giou_loss = tf.math.divide_no_nan(tf.math.reduce_sum(giou_loss),
            avg_factor)
    else:
        giou_loss = tf.math.divide_no_nan(giou_loss, avg_factor)
    assert giou_loss.dtype == tf.float32
    return giou_loss
