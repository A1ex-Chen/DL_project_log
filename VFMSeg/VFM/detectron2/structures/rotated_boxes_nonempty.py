def nonempty(self, threshold: float=0.0) ->torch.Tensor:
    """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor: a binary vector which represents
            whether each box is empty (False) or non-empty (True).
        """
    box = self.tensor
    widths = box[:, 2]
    heights = box[:, 3]
    keep = (widths > threshold) & (heights > threshold)
    return keep
