def similarity_fn(anchors, gt_boxes):
    anchors_rbv = anchors[:, [0, 1, 3, 4, 6]]
    gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, 6]]
    return self._region_similarity_calculator.compare(anchors_rbv, gt_boxes_rbv
        )
