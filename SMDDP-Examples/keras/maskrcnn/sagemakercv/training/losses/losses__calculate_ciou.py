def _calculate_ciou(b1, b2, mode='diou'):
    """
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
            encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
            in boxes are encoded as [y_min, x_min, y_max, x_max].
        mode: one of ['diou', 'ciou'], decided to calculate GIoU or IoU loss.
    Returns:
        GIoU loss float `Tensor`.
    """
    zero = tf.convert_to_tensor(0.0, b1.dtype)
    b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
    b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
    b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height
    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height
    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)
    if mode == 'iou':
        return iou
    enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
    enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
    enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
    enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
    enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
    enclose_area = enclose_width * enclose_height
    if mode == 'giou':
        giou = iou - tf.math.divide_no_nan(enclose_area - union_area,
            enclose_area)
        return giou
    diag_length = tf.linalg.norm([enclose_height, enclose_width])
    b1_center = tf.stack([(b1_ymin + b1_ymax) / 2.0, (b1_xmin + b1_xmax) / 2.0]
        )
    b2_center = tf.stack([(b2_ymin + b2_ymax) / 2.0, (b2_xmin + b2_xmax) / 2.0]
        )
    centers_dist = tf.linalg.norm([b1_center - b2_center])
    diou = iou - tf.math.divide_no_nan(centers_dist ** 2, diag_length ** 2)
    if mode == 'diou':
        return diou
    arctan = tf.atan(tf.math.divide_no_nan(b1_width, b1_height)) - tf.atan(tf
        .math.divide_no_nan(b2_width, b2_height))
    v = 4.0 * (arctan / math.pi) ** 2
    aspect_penalty_mask = tf.cast(iou > 0.5, b1.dtype)
    alpha = aspect_penalty_mask * tf.math.divide_no_nan(v, 1.0 - iou + v)
    ciou = diou - alpha * v
    return ciou
