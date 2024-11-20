def inference_single_image(self, anchors: List[Boxes], box_cls: List[torch.
    Tensor], box_delta: List[torch.Tensor], image_size: Tuple[int, int]):
    """
        Identical to :meth:`RetinaNet.inference_single_image.
        """
    pred = self._decode_multi_level_predictions(anchors, box_cls, box_delta,
        self.test_score_thresh, self.test_topk_candidates, image_size)
    keep = batched_nms(pred.pred_boxes.tensor, pred.scores, pred.
        pred_classes, self.test_nms_thresh)
    return pred[keep[:self.max_detections_per_image]]
