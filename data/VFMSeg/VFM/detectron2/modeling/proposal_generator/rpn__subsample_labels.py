def _subsample_labels(self, label):
    """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
    pos_idx, neg_idx = subsample_labels(label, self.batch_size_per_image,
        self.positive_fraction, 0)
    label.fill_(-1)
    label.scatter_(0, pos_idx, 1)
    label.scatter_(0, neg_idx, 0)
    return label
