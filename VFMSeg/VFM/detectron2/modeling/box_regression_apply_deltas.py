def apply_deltas(self, deltas, boxes):
    """
        Apply transformation `deltas` (dx1, dy1, dx2, dy2) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
    deltas = F.relu(deltas)
    boxes = boxes.to(deltas.dtype)
    ctr_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    ctr_y = 0.5 * (boxes[:, 1] + boxes[:, 3])
    if self.normalize_by_size:
        stride_w = boxes[:, 2] - boxes[:, 0]
        stride_h = boxes[:, 3] - boxes[:, 1]
        strides = torch.stack([stride_w, stride_h, stride_w, stride_h], axis=1)
        deltas = deltas * strides
    l = deltas[:, 0::4]
    t = deltas[:, 1::4]
    r = deltas[:, 2::4]
    b = deltas[:, 3::4]
    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, 0::4] = ctr_x[:, None] - l
    pred_boxes[:, 1::4] = ctr_y[:, None] - t
    pred_boxes[:, 2::4] = ctr_x[:, None] + r
    pred_boxes[:, 3::4] = ctr_y[:, None] + b
    return pred_boxes
