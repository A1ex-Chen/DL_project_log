def format_predictions(image_id, detection_boxes, detection_scores,
    detection_classes, rles):
    box_predictions = []
    mask_predictions = []
    detection_count = len(detection_scores)
    for i in range(detection_count):
        box_predictions.append({'image_id': int(image_id), 'category_id':
            int(detection_classes[i]), 'bbox': list(map(lambda x: float(
            round(x, 2)), detection_boxes[i])), 'score': float(
            detection_scores[i])})
        if rles:
            segmentation = {'size': rles[i]['size'], 'counts': rles[i][
                'counts'].decode()}
            mask_predictions.append({'image_id': int(image_id),
                'category_id': int(detection_classes[i]), 'score': float(
                detection_scores[i]), 'segmentation': segmentation})
    return box_predictions, mask_predictions
