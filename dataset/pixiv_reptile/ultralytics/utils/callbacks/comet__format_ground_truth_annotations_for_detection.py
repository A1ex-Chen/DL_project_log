def _format_ground_truth_annotations_for_detection(img_idx, image_path,
    batch, class_name_map=None):
    """Format ground truth annotations for detection."""
    indices = batch['batch_idx'] == img_idx
    bboxes = batch['bboxes'][indices]
    if len(bboxes) == 0:
        LOGGER.debug(
            f'COMET WARNING: Image: {image_path} has no bounding boxes labels')
        return None
    cls_labels = batch['cls'][indices].squeeze(1).tolist()
    if class_name_map:
        cls_labels = [str(class_name_map[label]) for label in cls_labels]
    original_image_shape = batch['ori_shape'][img_idx]
    resized_image_shape = batch['resized_shape'][img_idx]
    ratio_pad = batch['ratio_pad'][img_idx]
    data = []
    for box, label in zip(bboxes, cls_labels):
        box = _scale_bounding_box_to_original_image_shape(box,
            resized_image_shape, original_image_shape, ratio_pad)
        data.append({'boxes': [box], 'label': f'gt_{label}', 'score':
            _scale_confidence_score(1.0)})
    return {'name': 'ground_truth', 'data': data}
