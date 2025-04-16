def _decode_per_level_predictions(self, anchors: Boxes, pred_scores: Tensor,
    pred_deltas: Tensor, score_thresh: float, topk_candidates: int,
    image_size: Tuple[int, int]) ->Instances:
    """
        Decode boxes and classification predictions of one featuer level, by
        the following steps:
        1. filter the predictions based on score threshold and top K scores.
        2. transform the box regression outputs
        3. return the predicted scores, classes and boxes

        Args:
            anchors: Boxes, anchor for this feature level
            pred_scores: HxWxA,K
            pred_deltas: HxWxA,4

        Returns:
            Instances: with field "scores", "pred_boxes", "pred_classes".
        """
    keep_idxs = pred_scores > score_thresh
    pred_scores = pred_scores[keep_idxs]
    topk_idxs = torch.nonzero(keep_idxs)
    num_topk = min(topk_candidates, topk_idxs.size(0))
    pred_scores, idxs = pred_scores.topk(num_topk)
    topk_idxs = topk_idxs[idxs]
    anchor_idxs, classes_idxs = topk_idxs.unbind(dim=1)
    pred_boxes = self.box2box_transform.apply_deltas(pred_deltas[
        anchor_idxs], anchors.tensor[anchor_idxs])
    return Instances(image_size, pred_boxes=Boxes(pred_boxes), scores=
        pred_scores, pred_classes=classes_idxs)
