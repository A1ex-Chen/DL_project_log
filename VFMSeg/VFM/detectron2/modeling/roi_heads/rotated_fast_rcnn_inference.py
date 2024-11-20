def inference(self, predictions, proposals):
    """
        Returns:
            list[Instances]: same as `fast_rcnn_inference_rotated`.
            list[Tensor]: same as `fast_rcnn_inference_rotated`.
        """
    boxes = self.predict_boxes(predictions, proposals)
    scores = self.predict_probs(predictions, proposals)
    image_shapes = [x.image_size for x in proposals]
    return fast_rcnn_inference_rotated(boxes, scores, image_shapes, self.
        test_score_thresh, self.test_nms_thresh, self.test_topk_per_image)
