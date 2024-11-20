def __call__(self, match_quality_matrix):
    """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero`
                for selecting indices in :meth:`set_low_quality_matches_`).

        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates
                whether a prediction is a true or false positive or ignored
        """
    assert match_quality_matrix.dim() == 2
    if match_quality_matrix.numel() == 0:
        default_matches = match_quality_matrix.new_full((
            match_quality_matrix.size(1),), 0, dtype=torch.int64)
        default_match_labels = match_quality_matrix.new_full((
            match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8)
        return default_matches, default_match_labels
    assert torch.all(match_quality_matrix >= 0)
    matched_vals, matches = match_quality_matrix.max(dim=0)
    match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)
    for l, low, high in zip(self.labels, self.thresholds[:-1], self.
        thresholds[1:]):
        low_high = (matched_vals >= low) & (matched_vals < high)
        match_labels[low_high] = l
    if self.allow_low_quality_matches:
        self.set_low_quality_matches_(match_labels, match_quality_matrix)
    return matches, match_labels
