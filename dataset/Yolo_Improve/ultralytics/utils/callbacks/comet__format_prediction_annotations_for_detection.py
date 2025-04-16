def _format_prediction_annotations_for_detection(image_path, metadata,
    class_label_map=None):
    """Format YOLO predictions for object detection visualization."""
    stem = image_path.stem
    image_id = int(stem) if stem.isnumeric() else stem
    predictions = metadata.get(image_id)
    if not predictions:
        LOGGER.debug(
            f'COMET WARNING: Image: {image_path} has no bounding boxes predictions'
            )
        return None
    data = []
    for prediction in predictions:
        boxes = prediction['bbox']
        score = _scale_confidence_score(prediction['score'])
        cls_label = prediction['category_id']
        if class_label_map:
            cls_label = str(class_label_map[cls_label])
        data.append({'boxes': [boxes], 'label': cls_label, 'score': score})
    return {'name': 'prediction', 'data': data}
