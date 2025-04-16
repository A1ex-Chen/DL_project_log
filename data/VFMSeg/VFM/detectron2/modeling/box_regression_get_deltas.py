def get_deltas(self, src_boxes, target_boxes):
    """
        Get box regression transformation deltas (dx1, dy1, dx2, dy2) that can be used
        to transform the `src_boxes` into the `target_boxes`. That is, the relation
        ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true.
        The center of src must be inside target boxes.

        Args:
            src_boxes (Tensor): square source boxes, e.g., anchors
            target_boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
    assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
    assert isinstance(target_boxes, torch.Tensor), type(target_boxes)
    src_ctr_x = 0.5 * (src_boxes[:, 0] + src_boxes[:, 2])
    src_ctr_y = 0.5 * (src_boxes[:, 1] + src_boxes[:, 3])
    target_l = src_ctr_x - target_boxes[:, 0]
    target_t = src_ctr_y - target_boxes[:, 1]
    target_r = target_boxes[:, 2] - src_ctr_x
    target_b = target_boxes[:, 3] - src_ctr_y
    deltas = torch.stack((target_l, target_t, target_r, target_b), dim=1)
    if self.normalize_by_size:
        stride_w = src_boxes[:, 2] - src_boxes[:, 0]
        stride_h = src_boxes[:, 3] - src_boxes[:, 1]
        strides = torch.stack([stride_w, stride_h, stride_w, stride_h], axis=1)
        deltas = deltas / strides
    return deltas
