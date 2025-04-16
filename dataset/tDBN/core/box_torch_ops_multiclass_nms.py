def multiclass_nms(nms_func, boxes, scores, num_class, pre_max_size=None,
    post_max_size=None, score_thresh=0.0, iou_threshold=0.5):
    selected_per_class = []
    assert len(boxes.shape) == 3, 'bbox must have shape [N, num_cls, 7]'
    assert len(scores.shape) == 2, 'score must have shape [N, num_cls]'
    num_class = scores.shape[1]
    if not (boxes.shape[1] == scores.shape[1] or boxes.shape[1] == 1):
        raise ValueError(
            'tDBN dimension of boxes must be either 1 or equal to the tDBN dimension of scores'
            )
    num_boxes = boxes.shape[0]
    num_scores = scores.shape[0]
    num_classes = scores.shape[1]
    boxes_ids = range(num_classes) if boxes.shape[1] > 1 else [0] * num_classes
    for class_idx, boxes_idx in zip(range(num_classes), boxes_ids):
        class_scores = scores[:, class_idx]
        class_boxes = boxes[:, boxes_idx]
        if score_thresh > 0.0:
            class_scores_keep = torch.nonzero(class_scores >= score_thresh)
            if class_scores_keep.shape[0] != 0:
                class_scores_keep = class_scores_keep[:, 0]
            else:
                selected_per_class.append(None)
                continue
            class_scores = class_scores[class_scores_keep]
        if class_scores.shape[0] != 0:
            if score_thresh > 0.0:
                class_boxes = class_boxes[class_scores_keep]
            keep = nms_func(class_boxes, class_scores, pre_max_size,
                post_max_size, iou_threshold)
            if keep is not None:
                if score_thresh > 0.0:
                    selected_per_class.append(class_scores_keep[keep])
                else:
                    selected_per_class.append(keep)
            else:
                selected_per_class.append(None)
        else:
            selected_per_class.append(None)
    return selected_per_class
