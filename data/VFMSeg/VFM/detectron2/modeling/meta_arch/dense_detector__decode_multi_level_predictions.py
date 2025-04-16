def _decode_multi_level_predictions(self, anchors: List[Boxes], pred_scores:
    List[Tensor], pred_deltas: List[Tensor], score_thresh: float,
    topk_candidates: int, image_size: Tuple[int, int]) ->Instances:
    """
        Run `_decode_per_level_predictions` for all feature levels and concat the results.
        """
    predictions = [self._decode_per_level_predictions(anchors_i, box_cls_i,
        box_reg_i, self.test_score_thresh, self.test_topk_candidates,
        image_size) for box_cls_i, box_reg_i, anchors_i in zip(pred_scores,
        pred_deltas, anchors)]
    return predictions[0].cat(predictions)
