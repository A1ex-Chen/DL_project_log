def predict_proposals(self, anchors: List[Boxes], pred_objectness_logits:
    List[torch.Tensor], pred_anchor_deltas: List[torch.Tensor], image_sizes:
    List[Tuple[int, int]]):
    """
        Decode all the predicted box regression deltas to proposals. Find the top proposals
        by applying NMS and removing boxes that are too small.

        Returns:
            proposals (list[Instances]): list of N Instances. The i-th Instances
                stores post_nms_topk object proposals for image i, sorted by their
                objectness score in descending order.
        """
    with torch.no_grad():
        pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
        return find_top_rpn_proposals(pred_proposals,
            pred_objectness_logits, image_sizes, self.nms_thresh, self.
            pre_nms_topk[self.training], self.post_nms_topk[self.training],
            self.min_box_size, self.training)
