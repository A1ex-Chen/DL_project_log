def __init__(self, tensor: torch.Tensor):
    """
        Args:
            tensor: (N, M, M) mask tensor that defines the mask within each ROI.
        """
    if tensor.dim() != 3:
        raise ValueError('ROIMasks must take a masks of 3 dimension.')
    self.tensor = tensor
