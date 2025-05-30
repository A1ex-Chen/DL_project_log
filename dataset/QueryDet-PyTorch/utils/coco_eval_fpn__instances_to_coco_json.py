def _instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []
    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    has_fpn_layer = instances.has('fpn_layers')
    if has_fpn_layer:
        fpn_layers = instances.fpn_layers.tolist()
    has_mask = instances.has('pred_masks')
    if has_mask:
        rles = [mask_util.encode(np.array(mask[:, :, None], order='F',
            dtype='uint8'))[0] for mask in instances.pred_masks]
        for rle in rles:
            rle['counts'] = rle['counts'].decode('utf-8')
    has_keypoints = instances.has('pred_keypoints')
    if has_keypoints:
        keypoints = instances.pred_keypoints
    results = []
    for k in range(num_instance):
        result = {'image_id': img_id, 'category_id': classes[k], 'bbox':
            boxes[k], 'score': scores[k]}
        if has_fpn_layer:
            result['fpn_layer'] = fpn_layers[k]
        if has_mask:
            result['segmentation'] = rles[k]
        if has_keypoints:
            keypoints[k][:, :2] -= 0.5
            result['keypoints'] = keypoints[k].flatten().tolist()
        results.append(result)
    return results
