def _add_class_assignments(iou, gt_boxes, gt_labels):
    """Computes object category assignment for each box.

  Args:
    iou: a tensor for the iou matrix with a shape of
      [batch_size, K, MAX_NUM_INSTANCES]. K is the number of post-nms RoIs
      (i.e., rpn_post_nms_topn).
    gt_boxes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4].
      This tensor might have paddings with negative values. The coordinates
      of gt_boxes are in the pixel coordinates of the scaled image scale.
    gt_labels: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES]. This
      tensor might have paddings with a value of -1.
  Returns:
    max_boxes: a tensor with a shape of [batch_size, K, 4], representing
      the ground truth coordinates of each roi.
    max_classes: a int32 tensor with a shape of [batch_size, K], representing
      the ground truth class of each roi.
    max_overlap: a tensor with a shape of [batch_size, K], representing
      the maximum overlap of each roi.
    argmax_iou: a tensor with a shape of [batch_size, K], representing the iou
      argmax.
  """
    with tf.name_scope('add_class_assignments'):
        batch_size, _, _ = iou.get_shape().as_list()
        argmax_iou = tf.argmax(input=iou, axis=2, output_type=tf.int32)
        indices = tf.reshape(argmax_iou + tf.expand_dims(tf.range(
            batch_size) * tf.shape(input=gt_labels)[1], 1), shape=[-1])
        max_classes = tf.reshape(tf.gather(tf.reshape(gt_labels, [-1, 1]),
            indices), [batch_size, -1])
        max_overlap = tf.reduce_max(input_tensor=iou, axis=2)
        bg_mask = tf.equal(max_overlap, tf.zeros_like(max_overlap))
        max_classes = tf.where(bg_mask, tf.zeros_like(max_classes), max_classes
            )
        max_boxes = tf.reshape(tf.gather(tf.reshape(gt_boxes, [-1, 4]),
            indices), [batch_size, -1, 4])
        max_boxes = tf.where(tf.tile(tf.expand_dims(bg_mask, axis=2), [1, 1,
            4]), tf.zeros_like(max_boxes), max_boxes)
    return max_boxes, max_classes, max_overlap, argmax_iou
