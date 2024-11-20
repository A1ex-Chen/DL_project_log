def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512),
    aspect_ratios=(0.5, 1, 2), angles=(-90, -60, -30, 0, 30, 60, 90)):
    """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes, aspect_ratios, angles centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).

        Args:
            sizes (tuple[float]):
            aspect_ratios (tuple[float]]):
            angles (tuple[float]]):

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios) * len(angles), 5)
                storing anchor boxes in (x_ctr, y_ctr, w, h, angle) format.
        """
    anchors = []
    for size in sizes:
        area = size ** 2.0
        for aspect_ratio in aspect_ratios:
            w = math.sqrt(area / aspect_ratio)
            h = aspect_ratio * w
            anchors.extend([0, 0, w, h, a] for a in angles)
    return torch.tensor(anchors)
