def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor],
    proposals: List[Instances]):
    """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
    boxes = self.predict_boxes(predictions, proposals)
    scores = self.predict_probs(predictions, proposals)
    image_shapes = [x.image_size for x in proposals]
    return fast_rcnn_inference(boxes, scores, image_shapes, self.
        test_score_thresh, self.test_nms_thresh, self.test_topk_per_image)
