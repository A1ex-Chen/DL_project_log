def __init__(self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int]]):
    """
        Arguments:
            tensor (Tensor): of shape (N, H, W) or (N, C_1, ..., C_K, H, W) where K >= 1
            image_sizes (list[tuple[int, int]]): Each tuple is (h, w). It can
                be smaller than (H, W) due to padding.
        """
    self.tensor = tensor
    self.image_sizes = image_sizes
