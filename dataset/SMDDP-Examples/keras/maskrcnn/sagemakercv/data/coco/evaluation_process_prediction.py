def process_prediction(prediction):
    prediction.update({'detection_boxes': process_boxes(prediction[
        'image_info'], prediction['detection_boxes'])})
    batch_size = prediction['num_detections'].shape[0]
    box_predictions = []
    mask_predictions = []
    imgIds = []
    for i in range(batch_size):
        detection_boxes = prediction['detection_boxes'][i]
        detection_classes = prediction['detection_classes'][i]
        detection_scores = prediction['detection_scores'][i]
        source_id = prediction['source_ids'][i]
        detection_masks = prediction['detection_masks'][i
            ] if 'detection_masks' in prediction.keys() else None
        segments = generate_segmentation_from_masks(detection_masks,
            detection_boxes, int(prediction['image_info'][i][3]), int(
            prediction['image_info'][i][4])
            ) if detection_masks is not None else None
        rles = None
        if detection_masks is not None:
            rles = [maskUtils.encode(np.asfortranarray(instance_mask.astype
                (np.uint8))) for instance_mask in segments]
        formatted_predictions = format_predictions(source_id,
            detection_boxes, detection_scores, detection_classes, rles)
        imgIds.append(source_id)
        box_predictions.extend(formatted_predictions[0])
        mask_predictions.extend(formatted_predictions[1])
    return imgIds, box_predictions, mask_predictions
