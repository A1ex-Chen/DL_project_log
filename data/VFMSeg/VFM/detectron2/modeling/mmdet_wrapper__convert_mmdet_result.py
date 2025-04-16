def _convert_mmdet_result(result, shape: Tuple[int, int]) ->Instances:
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]
    else:
        bbox_result, segm_result = result, None
    bboxes = torch.from_numpy(np.vstack(bbox_result))
    bboxes, scores = bboxes[:, :4], bboxes[:, -1]
    labels = [torch.full((bbox.shape[0],), i, dtype=torch.int32) for i,
        bbox in enumerate(bbox_result)]
    labels = torch.cat(labels)
    inst = Instances(shape)
    inst.pred_boxes = Boxes(bboxes)
    inst.scores = scores
    inst.pred_classes = labels
    if segm_result is not None and len(labels) > 0:
        segm_result = list(itertools.chain(*segm_result))
        segm_result = [(torch.from_numpy(x) if isinstance(x, np.ndarray) else
            x) for x in segm_result]
        segm_result = torch.stack(segm_result, dim=0)
        inst.pred_masks = segm_result
    return inst
