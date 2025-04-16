def set_low_quality_matches_(self, match_labels, match_quality_matrix):
    """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.

        This function implements the RPN assignment case (i) in Sec. 3.1.2 of
        :paper:`Faster R-CNN`.
        """
    highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
    _, pred_inds_with_highest_quality = nonzero_tuple(match_quality_matrix ==
        highest_quality_foreach_gt[:, None])
    match_labels[pred_inds_with_highest_quality] = 1
