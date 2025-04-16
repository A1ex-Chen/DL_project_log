def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int=0
    ) ->torch.Tensor:
    """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
    height, width = box_size
    inds_inside = (self.tensor[..., 0] >= -boundary_threshold) & (self.
        tensor[..., 1] >= -boundary_threshold) & (self.tensor[..., 2] < 
        width + boundary_threshold) & (self.tensor[..., 3] < height +
        boundary_threshold)
    return inds_inside
