def __init__(self, matched_threshold, unmatched_threshold=None,
    negatives_lower_than_unmatched=True, force_match_for_each_row=False):
    """Construct ArgMaxMatcher.

    Args:
      matched_threshold: Threshold for positive matches. Positive if
        sim >= matched_threshold, where sim is the maximum value of the
        similarity matrix for a given column. Set to None for no threshold.
      unmatched_threshold: Threshold for negative matches. Negative if
        sim < unmatched_threshold. Defaults to matched_threshold
        when set to None.
      negatives_lower_than_unmatched: Boolean which defaults to True. If True
        then negative matches are the ones below the unmatched_threshold,
        whereas ignored matches are in between the matched and umatched
        threshold. If False, then negative matches are in between the matched
        and unmatched threshold, and everything lower than unmatched is ignored.
      force_match_for_each_row: If True, ensures that each row is matched to
        at least one column (which is not guaranteed otherwise if the
        matched_threshold is high). Defaults to False. See
        argmax_matcher_test.testMatcherForceMatch() for an example.

    Raises:
      ValueError: if unmatched_threshold is set but matched_threshold is not set
        or if unmatched_threshold > matched_threshold.
    """
    if matched_threshold is None and unmatched_threshold is not None:
        raise ValueError(
            'Need to also define matched_threshold whenunmatched_threshold is defined'
            )
    self._matched_threshold = matched_threshold
    if unmatched_threshold is None:
        self._unmatched_threshold = matched_threshold
    else:
        if unmatched_threshold > matched_threshold:
            raise ValueError(
                'unmatched_threshold needs to be smaller or equalto matched_threshold'
                )
        self._unmatched_threshold = unmatched_threshold
    if not negatives_lower_than_unmatched:
        if self._unmatched_threshold == self._matched_threshold:
            raise ValueError(
                'When negatives are in between matched and unmatched thresholds, these cannot be of equal value. matched: %s, unmatched: %s'
                , self._matched_threshold, self._unmatched_threshold)
    self._force_match_for_each_row = force_match_for_each_row
    self._negatives_lower_than_unmatched = negatives_lower_than_unmatched
