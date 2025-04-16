def get_mask_targets(fg_boxes, fg_proposal_to_label_map, fg_box_targets,
    mask_gt_labels, output_size=28):
    """Crop and resize on multilevel feature pyramid.

    Args:
    fg_boxes: A 3-D tensor of shape [batch_size, num_masks, 4]. Each row
      represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
    fg_proposal_to_label_map: A tensor of shape [batch_size, num_masks].
    fg_box_targets: a float tensor representing the box label for each box
      with a shape of [batch_size, num_masks, 4].
    mask_gt_labels: A tensor with a shape of [batch_size, M, H+4, W+4]. M is
      NUM_MAX_INSTANCES (i.e., 100 in this implementation) in each image, while
      H and W are ground truth mask size. The `+4` comes from padding of two
      zeros in both directions of height and width dimension.
    output_size: A scalar to indicate the output crop size.

    Returns:
    A 4-D tensor representing feature crop of shape
    [batch_size, num_boxes, output_size, output_size].
    """
    _, _, max_feature_height, max_feature_width = mask_gt_labels.get_shape(
        ).as_list()
    levels = tf.maximum(fg_proposal_to_label_map, 0)
    bb_y_min, bb_x_min, bb_y_max, bb_x_max = tf.split(value=fg_boxes,
        num_or_size_splits=4, axis=2)
    gt_y_min, gt_x_min, gt_y_max, gt_x_max = tf.split(value=fg_box_targets,
        num_or_size_splits=4, axis=2)
    valid_feature_width = max_feature_width - 4
    valid_feature_height = max_feature_height - 4
    y_transform = (bb_y_min - gt_y_min) * valid_feature_height / (gt_y_max -
        gt_y_min + _EPSILON) + 2
    x_transform = (bb_x_min - gt_x_min) * valid_feature_width / (gt_x_max -
        gt_x_min + _EPSILON) + 2
    h_transform = (bb_y_max - bb_y_min) * valid_feature_height / (gt_y_max -
        gt_y_min + _EPSILON)
    w_transform = (bb_x_max - bb_x_min) * valid_feature_width / (gt_x_max -
        gt_x_min + _EPSILON)
    boundaries = tf.concat([tf.cast(tf.ones_like(y_transform) * (
        max_feature_height - 1), dtype=tf.float32), tf.cast(tf.ones_like(
        x_transform) * (max_feature_width - 1), dtype=tf.float32)], axis=-1)
    features_per_box = spatial_transform_ops.selective_crop_and_resize(tf.
        expand_dims(mask_gt_labels, -1), tf.concat([y_transform,
        x_transform, h_transform, w_transform], -1), tf.expand_dims(levels,
        -1), boundaries, output_size)
    features_per_box = tf.squeeze(features_per_box, axis=-1)
    features_per_box = tf.where(tf.greater_equal(features_per_box, 0.5), tf
        .ones_like(features_per_box), tf.zeros_like(features_per_box))
    features_per_box = tf.stop_gradient(features_per_box)
    return features_per_box
