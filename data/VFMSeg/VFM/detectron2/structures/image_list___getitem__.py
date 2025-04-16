def __getitem__(self, idx) ->torch.Tensor:
    """
        Access the individual image in its original size.

        Args:
            idx: int or slice

        Returns:
            Tensor: an image of shape (H, W) or (C_1, ..., C_K, H, W) where K >= 1
        """
    size = self.image_sizes[idx]
    return self.tensor[idx, ..., :size[0], :size[1]]
