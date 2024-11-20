def __init__(self, anchors, num_classes, match_threshold=0.7,
    unmatched_threshold=0.3, rpn_batch_size_per_im=256, rpn_fg_fraction=0.5):
    """Constructs anchor labeler to assign labels to anchors.

    Args:
      anchors: an instance of class Anchors.
      num_classes: integer number representing number of classes in the dataset.
      match_threshold: a float number between 0 and 1 representing the
        lower-bound threshold to assign positive labels for anchors. An anchor
        with a score over the threshold is labeled positive.
      unmatched_threshold: a float number between 0 and 1 representing the
        upper-bound threshold to assign negative labels for anchors. An anchor
        with a score below the threshold is labeled negative.
      rpn_batch_size_per_im: a integer number that represents the number of
        sampled anchors per image in the first stage (region proposal network).
      rpn_fg_fraction: a float number between 0 and 1 representing the fraction
        of positive anchors (foreground) in the first stage.
    """
    similarity_calc = region_similarity_calculator.IouSimilarity()
    matcher = argmax_matcher.ArgMaxMatcher(match_threshold,
        unmatched_threshold=unmatched_threshold,
        negatives_lower_than_unmatched=True, force_match_for_each_row=True)
    box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()
    self._target_assigner = target_assigner.TargetAssigner(similarity_calc,
        matcher, box_coder)
    self._anchors = anchors
    self._match_threshold = match_threshold
    self._unmatched_threshold = unmatched_threshold
    self._rpn_batch_size_per_im = rpn_batch_size_per_im
    self._rpn_fg_fraction = rpn_fg_fraction
    self._num_classes = num_classes
