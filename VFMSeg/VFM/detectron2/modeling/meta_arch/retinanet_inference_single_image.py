def inference_single_image(self, anchors: List[Boxes], box_cls: List[Tensor
    ], box_delta: List[Tensor], image_size: Tuple[int, int]):
    """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature level.
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
    pred = self._decode_multi_level_predictions(anchors, box_cls, box_delta,
        self.test_score_thresh, self.test_topk_candidates, image_size)
    keep = batched_nms(pred.pred_boxes.tensor, pred.scores, pred.
        pred_classes, self.test_nms_thresh)
    return pred[keep[:self.max_detections_per_image]]
