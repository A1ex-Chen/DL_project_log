def encode_box_targets(boxes, gt_boxes, gt_labels, bbox_reg_weights):
    """Encodes predicted boxes with respect to ground truth boxes."""
    with tf.name_scope('encode_box_targets'):
        box_targets = box_utils.encode_boxes(boxes=gt_boxes, anchors=boxes,
            weights=bbox_reg_weights)
        mask = tf.tile(tf.expand_dims(tf.equal(gt_labels, tf.zeros_like(
            gt_labels)), axis=2), [1, 1, 4])
        box_targets = tf.where(mask, tf.zeros_like(box_targets), box_targets)
    return box_targets
