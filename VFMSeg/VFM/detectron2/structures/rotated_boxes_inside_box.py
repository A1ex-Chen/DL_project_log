def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int=0
    ) ->torch.Tensor:
    """
        Args:
            box_size (height, width): Size of the reference box covering
                [0, width] x [0, height]
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        For RRPN, it might not be necessary to call this function since it's common
        for rotated box to extend to outside of the image boundaries
        (the clip function only clips the near-horizontal boxes)

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
    height, width = box_size
    cnt_x = self.tensor[..., 0]
    cnt_y = self.tensor[..., 1]
    half_w = self.tensor[..., 2] / 2.0
    half_h = self.tensor[..., 3] / 2.0
    a = self.tensor[..., 4]
    c = torch.abs(torch.cos(a * math.pi / 180.0))
    s = torch.abs(torch.sin(a * math.pi / 180.0))
    max_rect_dx = c * half_w + s * half_h
    max_rect_dy = c * half_h + s * half_w
    inds_inside = (cnt_x - max_rect_dx >= -boundary_threshold) & (cnt_y -
        max_rect_dy >= -boundary_threshold) & (cnt_x + max_rect_dx < width +
        boundary_threshold) & (cnt_y + max_rect_dy < height +
        boundary_threshold)
    return inds_inside
