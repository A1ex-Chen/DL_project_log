def assemble_rcnn_outputs_by_name(image_sizes, tensor_outputs,
    force_mask_on=False):
    """
    A function to assemble caffe2 model's outputs (i.e. Dict[str, Tensor])
    to detectron2's format (i.e. list of Instances instance).
    This only works when the model follows the Caffe2 detectron's naming convention.

    Args:
        image_sizes (List[List[int, int]]): [H, W] of every image.
        tensor_outputs (Dict[str, Tensor]): external_output to its tensor.

        force_mask_on (Bool): if true, the it make sure there'll be pred_masks even
            if the mask is not found from tensor_outputs (usually due to model crash)
    """
    results = [Instances(image_size) for image_size in image_sizes]
    batch_splits = tensor_outputs.get('batch_splits', None)
    if batch_splits:
        raise NotImplementedError()
    assert len(image_sizes) == 1
    result = results[0]
    bbox_nms = tensor_outputs['bbox_nms']
    score_nms = tensor_outputs['score_nms']
    class_nms = tensor_outputs['class_nms']
    assert bbox_nms is not None
    assert score_nms is not None
    assert class_nms is not None
    if bbox_nms.shape[1] == 5:
        result.pred_boxes = RotatedBoxes(bbox_nms)
    else:
        result.pred_boxes = Boxes(bbox_nms)
    result.scores = score_nms
    result.pred_classes = class_nms.to(torch.int64)
    mask_fcn_probs = tensor_outputs.get('mask_fcn_probs', None)
    if mask_fcn_probs is not None:
        mask_probs_pred = mask_fcn_probs
        num_masks = mask_probs_pred.shape[0]
        class_pred = result.pred_classes
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = mask_probs_pred[indices, class_pred][:, None]
        result.pred_masks = mask_probs_pred
    elif force_mask_on:
        result.pred_masks = torch.zeros([0, 1, 0, 0], dtype=torch.uint8)
    keypoints_out = tensor_outputs.get('keypoints_out', None)
    kps_score = tensor_outputs.get('kps_score', None)
    if keypoints_out is not None:
        keypoints_tensor = keypoints_out
        keypoint_xyp = keypoints_tensor.transpose(1, 2)[:, :, [0, 1, 2]]
        result.pred_keypoints = keypoint_xyp
    elif kps_score is not None:
        pred_keypoint_logits = kps_score
        keypoint_head.keypoint_rcnn_inference(pred_keypoint_logits, [result])
    return results
