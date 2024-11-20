def nms(bboxes, score_threshold, iou_threshold, sigma=0.3, method='nms',
    max_dets=300):
    classes_in_img = list(set(bboxes[:, 5].astype(np.int32)))
    best_bboxes = []
    for cls in classes_in_img:
        cls_mask = bboxes[:, 5].astype(np.int32) == cls
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[:max_ind], cls_bboxes[
                max_ind + 1:]])
            iou = iou_xyxy_numpy(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            assert method in ['nms', 'soft-nms']
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > score_threshold
            cls_bboxes = cls_bboxes[score_mask]
    best_bboxes = np.array(best_bboxes)
    if len(best_bboxes) > max_dets:
        best_bboxes = best_bboxes[:max_dets, :]
    return best_bboxes
