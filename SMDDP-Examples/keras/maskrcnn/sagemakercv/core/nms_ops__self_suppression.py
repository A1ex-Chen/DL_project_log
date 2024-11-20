def _self_suppression(iou, _, iou_sum):
    batch_size = tf.shape(iou)[0]
    can_suppress_others = tf.cast(tf.reshape(tf.reduce_max(iou, 1) <= 0.5,
        [batch_size, -1, 1]), iou.dtype)
    iou_suppressed = tf.reshape(tf.cast(tf.reduce_max(can_suppress_others *
        iou, 1) <= 0.5, iou.dtype), [batch_size, -1, 1]) * iou
    iou_sum_new = tf.reduce_sum(iou_suppressed, [1, 2])
    return [iou_suppressed, tf.reduce_any(iou_sum - iou_sum_new > 0.5),
        iou_sum_new]
