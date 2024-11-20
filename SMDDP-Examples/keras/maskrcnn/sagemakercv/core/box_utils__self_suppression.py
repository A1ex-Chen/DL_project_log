def _self_suppression(iou, _, iou_sum):
    batch_size = tf.shape(input=iou)[0]
    can_suppress_others = tf.cast(tf.reshape(tf.reduce_max(input_tensor=iou,
        axis=1) <= 0.5, [batch_size, -1, 1]), iou.dtype)
    iou_suppressed = tf.reshape(tf.cast(tf.reduce_max(input_tensor=
        can_suppress_others * iou, axis=1) <= 0.5, iou.dtype), [batch_size,
        -1, 1]) * iou
    iou_sum_new = tf.reduce_sum(input_tensor=iou_suppressed, axis=[1, 2])
    return [iou_suppressed, tf.reduce_any(input_tensor=iou_sum -
        iou_sum_new > 0.5), iou_sum_new]
