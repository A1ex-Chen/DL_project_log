def fast_rcnn_inference_single_image_rotated(boxes, scores, image_shape,
    score_thresh, nms_thresh, topk_per_image):
    """
    Single-image inference. Return rotated bounding-box detection results by thresholding
    on scores and applying rotated non-maximum suppression (Rotated NMS).

    Args:
        Same as `fast_rcnn_inference_rotated`, but with rotated boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference_rotated`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(
        dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
    B = 5
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // B
    boxes = RotatedBoxes(boxes.reshape(-1, B))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, B)
    filter_mask = scores > score_thresh
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    keep = batched_nms_rotated(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    result = Instances(image_shape)
    result.pred_boxes = RotatedBoxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]
