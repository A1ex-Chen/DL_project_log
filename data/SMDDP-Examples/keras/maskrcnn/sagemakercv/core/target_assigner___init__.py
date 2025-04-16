def __init__(self, similarity_calc, matcher, box_coder,
    negative_class_weight=1.0, unmatched_cls_target=None):
    """Construct Object Detection Target Assigner.

        Args:
          similarity_calc: a RegionSimilarityCalculator
          matcher: Matcher used to match groundtruth to anchors.
          box_coder: BoxCoder used to encode matching groundtruth boxes with
            respect to anchors.
          negative_class_weight: classification weight to be associated to negative
            anchors (default: 1.0). The weight must be in [0., 1.].
          unmatched_cls_target: a float32 tensor with shape [d_1, d_2, ..., d_k]
            which is consistent with the classification target for each
            anchor (and can be empty for scalar targets).  This shape must thus be
            compatible with the groundtruth labels that are passed to the "assign"
            function (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]).
            If set to None, unmatched_cls_target is set to be [0] for each anchor.

        Raises:
          ValueError: if similarity_calc is not a RegionSimilarityCalculator or
            if matcher is not a Matcher or if box_coder is not a BoxCoder
        """
    self._similarity_calc = similarity_calc
    self._matcher = matcher
    self._box_coder = box_coder
    self._negative_class_weight = negative_class_weight
    if unmatched_cls_target is None:
        self._unmatched_cls_target = tf.constant([0], tf.float32)
    else:
        self._unmatched_cls_target = unmatched_cls_target
