def iou_scores(bbox_dets: (list[int] | list[list[int]]), bbox_gts: (list[
    int] | list[list[int]]), xywh: bool=False, average: bool=True) ->list[float
    ]:
    if isinstance(bbox_dets[0], int):
        bbox_dets = [bbox_dets]
        bbox_gts = [bbox_gts]
    assert len(bbox_dets) == len(bbox_gts
        ), f'The number of detected ({len(bbox_dets)}) and ground truth ({len(bbox_gts)} bounding boxes must be equal.'
    if xywh:
        bbox_dets = [_convert_xywh_to_x1y1x2y2(bbox_det) for bbox_det in
            bbox_dets]
        bbox_gts = [_convert_xywh_to_x1y1x2y2(bbox_gt) for bbox_gt in bbox_gts]
    scores = [iou_score(bbox_det, bbox_gt) for bbox_det, bbox_gt in zip(
        bbox_dets, bbox_gts)]
    if average:
        return sum(scores) / len(scores)
    return scores
