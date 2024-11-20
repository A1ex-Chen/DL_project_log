def inference_single_image(self, box_cls, box_delta, anchors, image_size):
    """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
    boxes_all = []
    scores_all = []
    class_idxs_all = []
    for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
        box_cls_i = box_cls_i.flatten().sigmoid_()
        num_topk = min(self.topk_candidates, box_reg_i.size(0))
        predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
        predicted_prob = predicted_prob[:num_topk]
        topk_idxs = topk_idxs[:num_topk]
        keep_idxs = predicted_prob > self.score_threshold
        predicted_prob = predicted_prob[keep_idxs]
        topk_idxs = topk_idxs[keep_idxs]
        anchor_idxs = topk_idxs // self.num_classes
        classes_idxs = topk_idxs % self.num_classes
        box_reg_i = box_reg_i[anchor_idxs]
        anchors_i = anchors_i[anchor_idxs]
        predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i,
            anchors_i.tensor)
        boxes_all.append(predicted_boxes)
        scores_all.append(predicted_prob)
        class_idxs_all.append(classes_idxs)
    boxes_all, scores_all, class_idxs_all = [cat(x) for x in [boxes_all,
        scores_all, class_idxs_all]]
    keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.
        nms_threshold)
    keep = keep[:self.max_detections_per_image]
    result = Instances(image_size)
    result.pred_boxes = Boxes(boxes_all[keep])
    result.scores = scores_all[keep]
    result.pred_classes = class_idxs_all[keep]
    return result
