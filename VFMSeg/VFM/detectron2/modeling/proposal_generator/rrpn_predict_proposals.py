@torch.no_grad()
def predict_proposals(self, anchors, pred_objectness_logits,
    pred_anchor_deltas, image_sizes):
    pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
    return find_top_rrpn_proposals(pred_proposals, pred_objectness_logits,
        image_sizes, self.nms_thresh, self.pre_nms_topk[self.training],
        self.post_nms_topk[self.training], self.min_box_size, self.training)
