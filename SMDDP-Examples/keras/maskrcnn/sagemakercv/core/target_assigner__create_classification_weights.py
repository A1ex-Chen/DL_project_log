def _create_classification_weights(self, match, groundtruth_weights):
    """Create classification weights for each anchor.

        Positive (matched) anchors are associated with a weight of
        positive_class_weight and negative (unmatched) anchors are associated with
        a weight of negative_class_weight. When anchors are ignored, weights are set
        to zero. By default, both positive/negative weights are set to 1.0,
        but they can be adjusted to handle class imbalance (which is almost always
        the case in object detection).

        Args:
          match: a matcher.Match object that provides a matching between anchors
            and groundtruth boxes.
          groundtruth_weights: a float tensor of shape [M] indicating the weight to
            assign to all anchors match to a particular groundtruth box.

        Returns:
          a float32 tensor with shape [num_anchors] representing classification
          weights.
        """
    return match.gather_based_on_match(groundtruth_weights, ignored_value=
        0.0, unmatched_value=self._negative_class_weight)
