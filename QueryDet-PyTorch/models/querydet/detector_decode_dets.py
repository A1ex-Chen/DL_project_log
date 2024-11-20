def decode_dets(self, cls_results, reg_results, anchors):
    boxes_all = []
    scores_all = []
    class_idxs_all = []
    for cls_i, reg_i, anchors_i in zip(cls_results, reg_results, anchors):
        cls_i = cls_i.view(-1, self.num_classes)
        reg_i = reg_i.view(-1, 4)
        cls_i = cls_i.flatten().sigmoid_()
        num_topk = min(self.topk_candidates, reg_i.size(0))
        predicted_prob, topk_idxs = cls_i.sort(descending=True)
        predicted_prob = predicted_prob[:num_topk]
        topk_idxs = topk_idxs[:num_topk]
        keep_idxs = predicted_prob > self.score_threshold
        predicted_prob = predicted_prob[keep_idxs]
        topk_idxs = topk_idxs[keep_idxs]
        anchor_idxs = topk_idxs // self.num_classes
        classes_idxs = topk_idxs % self.num_classes
        predicted_class = classes_idxs
        reg_i = reg_i[anchor_idxs]
        anchors_i = anchors_i[anchor_idxs]
        if type(anchors_i) != torch.Tensor:
            anchors_i = anchors_i.tensor
        predicted_boxes = self.box2box_transform.apply_deltas(reg_i, anchors_i)
        boxes_all.append(predicted_boxes)
        scores_all.append(predicted_prob)
        class_idxs_all.append(predicted_class)
    return boxes_all, scores_all, class_idxs_all
