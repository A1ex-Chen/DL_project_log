def _create_regression_weights(self, match, groundtruth_weights):
    """Set regression weight for each anchor.

        Only positive anchors are set to contribute to the regression loss, so this
        method returns a weight of 1 for every positive anchor and 0 for every
        negative anchor.

        Args:
          match: a matcher.Match object that provides a matching between anchors
            and groundtruth boxes.
          groundtruth_weights: a float tensor of shape [M] indicating the weight to
            assign to all anchors match to a particular groundtruth box.

        Returns:
          a float32 tensor with shape [num_anchors] representing regression weights.
        """
    return match.gather_based_on_match(groundtruth_weights, ignored_value=
        0.0, unmatched_value=0.0)
