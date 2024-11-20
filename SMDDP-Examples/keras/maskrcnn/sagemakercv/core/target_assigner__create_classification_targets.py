def _create_classification_targets(self, groundtruth_labels, match):
    """Create classification targets for each anchor.

        Assign a classification target of for each anchor to the matching
        groundtruth label that is provided by match.  Anchors that are not matched
        to anything are given the target self._unmatched_cls_target

        Args:
          groundtruth_labels:  a tensor of shape [num_gt_boxes, d_1, ... d_k]
            with labels for each of the ground_truth boxes. The subshape
            [d_1, ... d_k] can be empty (corresponding to scalar labels).
          match: a matcher.Match object that provides a matching between anchors
            and groundtruth boxes.

        Returns:
          a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k], where the
          subshape [d_1, ..., d_k] is compatible with groundtruth_labels which has
          shape [num_gt_boxes, d_1, d_2, ... d_k].
        """
    return match.gather_based_on_match(groundtruth_labels, unmatched_value=
        self._unmatched_cls_target, ignored_value=self._unmatched_cls_target)
