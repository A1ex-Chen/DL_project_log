def _create_regression_targets(self, anchors, groundtruth_boxes, match):
    """Returns a regression target for each anchor.

        Args:
          anchors: a BoxList representing N anchors
          groundtruth_boxes: a BoxList representing M groundtruth_boxes
          match: a matcher.Match object

        Returns:
          reg_targets: a float32 tensor with shape [N, box_code_dimension]
        """
    matched_gt_boxes = match.gather_based_on_match(groundtruth_boxes.get(),
        unmatched_value=tf.zeros(4), ignored_value=tf.zeros(4))
    matched_gt_boxlist = box_list.BoxList(matched_gt_boxes)
    if groundtruth_boxes.has_field(KEYPOINTS_FIELD_NAME):
        groundtruth_keypoints = groundtruth_boxes.get_field(
            KEYPOINTS_FIELD_NAME)
        matched_keypoints = match.gather_based_on_match(groundtruth_keypoints,
            unmatched_value=tf.zeros(groundtruth_keypoints.get_shape()[1:]),
            ignored_value=tf.zeros(groundtruth_keypoints.get_shape()[1:]))
        matched_gt_boxlist.add_field(KEYPOINTS_FIELD_NAME, matched_keypoints)
    matched_reg_targets = self._box_coder.encode(matched_gt_boxlist, anchors)
    match_results_shape = shape_utils.combined_static_and_dynamic_shape(match
        .match_results)
    unmatched_ignored_reg_targets = tf.tile(self._default_regression_target
        (), [match_results_shape[0], 1])
    matched_anchors_mask = match.matched_column_indicator()
    matched_anchors_mask = tf.expand_dims(matched_anchors_mask, axis=1)
    matched_anchors_mask = tf.broadcast_to(matched_anchors_mask, shape=
        matched_reg_targets.get_shape())
    reg_targets = tf.where(matched_anchors_mask, matched_reg_targets,
        unmatched_ignored_reg_targets)
    return reg_targets
