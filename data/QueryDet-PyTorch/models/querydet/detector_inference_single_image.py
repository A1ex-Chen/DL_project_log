def inference_single_image(self, retina_box_cls, retina_box_delta,
    retina_anchors, small_det_logits, small_det_delta, small_det_anchors,
    image_size):
    with autocast(False):
        all_cls = small_det_logits + retina_box_cls
        all_delta = small_det_delta + retina_box_delta
        all_anchors = small_det_anchors + retina_anchors
        boxes_all, scores_all, class_idxs_all = self.decode_dets(all_cls,
            all_delta, all_anchors)
        boxes_all, scores_all, class_idxs_all = [cat(x) for x in [boxes_all,
            scores_all, class_idxs_all]]
        if self.use_soft_nms:
            keep, soft_nms_scores = self.soft_nmser(boxes_all, scores_all,
                class_idxs_all)
        else:
            keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.
                nms_threshold)
        result = Instances(image_size)
        keep = keep[:self.max_detections_per_image]
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result
