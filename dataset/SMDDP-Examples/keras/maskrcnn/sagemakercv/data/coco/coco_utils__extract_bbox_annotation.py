def _extract_bbox_annotation(prediction, b, obj_i):
    """Constructs COCO format bounding box annotation."""
    height = prediction['height'][b]
    width = prediction['width'][b]
    bbox = _denormalize_to_coco_bbox(prediction['groundtruth_boxes'][b][
        obj_i, :], height, width)
    if 'groundtruth_area' in prediction:
        area = float(prediction['groundtruth_area'][b][obj_i])
    else:
        area = bbox[2] * bbox[3]
    annotation = {'id': b * 1000 + obj_i, 'image_id': int(prediction[
        'source_id'][b]), 'category_id': int(prediction[
        'groundtruth_classes'][b][obj_i]), 'bbox': bbox, 'iscrowd': int(
        prediction['groundtruth_is_crowd'][b][obj_i]), 'area': area,
        'segmentation': []}
    return annotation
