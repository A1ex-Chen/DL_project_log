def crop_and_resize(self, boxes: torch.Tensor, mask_size: int) ->torch.Tensor:
    """
        Crop each mask by the given box, and resize results to (mask_size, mask_size).
        This can be used to prepare training targets for Mask R-CNN.

        Args:
            boxes (Tensor): Nx4 tensor storing the boxes for each mask
            mask_size (int): the size of the rasterized mask.

        Returns:
            Tensor: A bool tensor of shape (N, mask_size, mask_size), where
            N is the number of predicted boxes for this image.
        """
    assert len(boxes) == len(self), '{} != {}'.format(len(boxes), len(self))
    device = boxes.device
    boxes = boxes.to(torch.device('cpu'))
    results = [rasterize_polygons_within_box(poly, box.numpy(), mask_size) for
        poly, box in zip(self.polygons, boxes)]
    """
        poly: list[list[float]], the polygons for one instance
        box: a tensor of shape (4,)
        """
    if len(results) == 0:
        return torch.empty(0, mask_size, mask_size, dtype=torch.bool,
            device=device)
    return torch.stack(results, dim=0).to(device=device)
