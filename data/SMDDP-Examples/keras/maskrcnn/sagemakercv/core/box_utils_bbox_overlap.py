def bbox_overlap(boxes, gt_boxes):
    """Calculates the overlap between proposal and ground truth boxes.

  Some `gt_boxes` may have been padded.  The returned `iou` tensor for these
  boxes will be -1.

  Args:
    boxes: a tensor with a shape of [batch_size, N, 4]. N is the number of
      proposals before groundtruth assignment (e.g., rpn_post_nms_topn). The
      last dimension is the pixel coordinates in [ymin, xmin, ymax, xmax] form.
    gt_boxes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4]. This
      tensor might have paddings with a negative value.
  Returns:
    iou: a tensor with as a shape of [batch_size, N, MAX_NUM_INSTANCES].
  """
    with tf.name_scope('bbox_overlap'):
        bb_y_min, bb_x_min, bb_y_max, bb_x_max = tf.split(value=boxes,
            num_or_size_splits=4, axis=2)
        gt_y_min, gt_x_min, gt_y_max, gt_x_max = tf.split(value=gt_boxes,
            num_or_size_splits=4, axis=2)
        i_xmin = tf.maximum(bb_x_min, tf.transpose(gt_x_min, [0, 2, 1]))
        i_xmax = tf.minimum(bb_x_max, tf.transpose(gt_x_max, [0, 2, 1]))
        i_ymin = tf.maximum(bb_y_min, tf.transpose(gt_y_min, [0, 2, 1]))
        i_ymax = tf.minimum(bb_y_max, tf.transpose(gt_y_max, [0, 2, 1]))
        i_area = tf.maximum(i_xmax - i_xmin, 0) * tf.maximum(i_ymax - i_ymin, 0
            )
        bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
        gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)
        u_area = bb_area + tf.transpose(gt_area, [0, 2, 1]) - i_area + 1e-08
        iou = i_area / u_area
        padding_mask = tf.less(i_xmin, tf.zeros_like(i_xmin))
        iou = tf.where(padding_mask, -tf.ones_like(iou), iou)
        return iou
