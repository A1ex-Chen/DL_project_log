def load_predictions(self, detection_results, include_mask, is_image_mask=
    False, log=True):
    """Create prediction dictionary list from detection and mask results.

    Args:
      detection_results: a dictionary containing numpy arrays which corresponds
        to prediction results.
      include_mask: a boolean, whether to include mask in detection results.
      is_image_mask: a boolean, where the predict mask is a whole image mask.

    Returns:
      a list of dictionary including different prediction results from the model
        in numpy form.
    """
    predictions = []
    num_detections = detection_results['detection_scores'].size
    current_index = 0
    for i, image_id in enumerate(detection_results['source_id']):
        if include_mask:
            box_coorindates_in_image = detection_results['detection_boxes'][i]
            segments = generate_segmentation_from_masks(detection_results[
                'detection_masks'][i], box_coorindates_in_image, int(
                detection_results['image_info'][i][3]), int(
                detection_results['image_info'][i][4]), is_image_mask=
                is_image_mask)
            encoded_masks = [maskUtils.encode(np.asfortranarray(
                instance_mask.astype(np.uint8))) for instance_mask in segments]
        for box_index in range(int(detection_results['num_detections'][i])):
            if current_index % 1000 == 0 and log:
                logging.info('{}/{}'.format(current_index, num_detections))
            current_index += 1
            prediction = {'image_id': int(image_id), 'bbox':
                detection_results['detection_boxes'][i][box_index].tolist(),
                'score': detection_results['detection_scores'][i][box_index
                ], 'category_id': int(detection_results['detection_classes'
                ][i][box_index])}
            if include_mask:
                prediction['segmentation'] = encoded_masks[box_index]
            predictions.append(prediction)
    return predictions
