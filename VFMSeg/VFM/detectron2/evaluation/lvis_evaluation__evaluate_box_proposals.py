def _evaluate_box_proposals(dataset_predictions, lvis_api, thresholds=None,
    area='all', limit=None):
    """
    Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official LVIS API recall evaluation code. However,
    it produces slightly different results.
    """
    areas = {'all': 0, 'small': 1, 'medium': 2, 'large': 3, '96-128': 4,
        '128-256': 5, '256-512': 6, '512-inf': 7}
    area_ranges = [[0 ** 2, 100000.0 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 **
        2], [96 ** 2, 100000.0 ** 2], [96 ** 2, 128 ** 2], [128 ** 2, 256 **
        2], [256 ** 2, 512 ** 2], [512 ** 2, 100000.0 ** 2]]
    assert area in areas, 'Unknown area range: {}'.format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0
    for prediction_dict in dataset_predictions:
        predictions = prediction_dict['proposals']
        inds = predictions.objectness_logits.sort(descending=True)[1]
        predictions = predictions[inds]
        ann_ids = lvis_api.get_ann_ids(img_ids=[prediction_dict['image_id']])
        anno = lvis_api.load_anns(ann_ids)
        gt_boxes = [BoxMode.convert(obj['bbox'], BoxMode.XYWH_ABS, BoxMode.
            XYXY_ABS) for obj in anno]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)
        gt_boxes = Boxes(gt_boxes)
        gt_areas = torch.as_tensor([obj['area'] for obj in anno])
        if len(gt_boxes) == 0 or len(predictions) == 0:
            continue
        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <=
            area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]
        num_pos += len(gt_boxes)
        if len(gt_boxes) == 0:
            continue
        if limit is not None and len(predictions) > limit:
            predictions = predictions[:limit]
        overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)
        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(predictions), len(gt_boxes))):
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            box_ind = argmax_overlaps[gt_ind]
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0) if len(gt_overlaps
        ) else torch.zeros(0, dtype=torch.float32)
    gt_overlaps, _ = torch.sort(gt_overlaps)
    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-05, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    ar = recalls.mean()
    return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
        'gt_overlaps': gt_overlaps, 'num_pos': num_pos}
