def forward(self, x: List[torch.Tensor], box_lists: List[Boxes]):
    """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.

        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
    num_level_assignments = len(self.level_poolers)
    assert isinstance(x, list) and isinstance(box_lists, list
        ), 'Arguments to pooler must be lists'
    assert len(x
        ) == num_level_assignments, 'unequal value, num_level_assignments={}, but x is list of {} Tensors'.format(
        num_level_assignments, len(x))
    assert len(box_lists) == x[0].size(0
        ), 'unequal value, x[0] batch dim 0 is {}, but box_list has length {}'.format(
        x[0].size(0), len(box_lists))
    if len(box_lists) == 0:
        return torch.zeros((0, x[0].shape[1]) + self.output_size, device=x[
            0].device, dtype=x[0].dtype)
    pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
    if num_level_assignments == 1:
        return self.level_poolers[0](x[0], pooler_fmt_boxes)
    level_assignments = assign_boxes_to_levels(box_lists, self.min_level,
        self.max_level, self.canonical_box_size, self.canonical_level)
    num_boxes = pooler_fmt_boxes.size(0)
    num_channels = x[0].shape[1]
    output_size = self.output_size[0]
    dtype, device = x[0].dtype, x[0].device
    output = torch.zeros((num_boxes, num_channels, output_size, output_size
        ), dtype=dtype, device=device)
    for level, pooler in enumerate(self.level_poolers):
        inds = nonzero_tuple(level_assignments == level)[0]
        pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
        output.index_put_((inds,), pooler(x[level], pooler_fmt_boxes_level))
    return output
