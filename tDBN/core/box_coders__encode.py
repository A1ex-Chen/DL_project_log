def _encode(self, boxes, anchors):
    anchors = anchors[..., [0, 1, 3, 4, 6]]
    boxes = boxes[..., [0, 1, 3, 4, 6]]
    return box_np_ops.bev_box_encode(boxes, anchors, self.vec_encode, self.
        linear_dim)
