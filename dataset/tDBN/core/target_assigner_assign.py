def assign(self, anchors, gt_boxes, anchors_mask=None, gt_classes=None,
    matched_thresholds=None, unmatched_thresholds=None):
    if anchors_mask is not None:
        prune_anchor_fn = lambda _: np.where(anchors_mask)[0]
    else:
        prune_anchor_fn = None

    def similarity_fn(anchors, gt_boxes):
        anchors_rbv = anchors[:, [0, 1, 3, 4, 6]]
        gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, 6]]
        return self._region_similarity_calculator.compare(anchors_rbv,
            gt_boxes_rbv)

    def box_encoding_fn(boxes, anchors):
        return self._box_coder.encode(boxes, anchors)
    return create_target_np(anchors, gt_boxes, similarity_fn,
        box_encoding_fn, prune_anchor_fn=prune_anchor_fn, gt_classes=
        gt_classes, matched_threshold=matched_thresholds,
        unmatched_threshold=unmatched_thresholds, positive_fraction=self.
        _positive_fraction, rpn_batch_size=self._sample_size,
        norm_by_num_examples=False, box_code_size=self.box_coder.code_size)
