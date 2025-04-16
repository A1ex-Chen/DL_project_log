def _merge_detections(self, all_boxes, all_scores, all_classes, shape_hw):
    num_boxes = len(all_boxes)
    num_classes = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
    all_scores_2d = torch.zeros(num_boxes, num_classes + 1, device=
        all_boxes.device)
    for idx, cls, score in zip(count(), all_classes, all_scores):
        all_scores_2d[idx, cls] = score
    merged_instances, _ = fast_rcnn_inference_single_image(all_boxes,
        all_scores_2d, shape_hw, 1e-08, self.cfg.MODEL.ROI_HEADS.
        NMS_THRESH_TEST, self.cfg.TEST.DETECTIONS_PER_IMAGE)
    return merged_instances
