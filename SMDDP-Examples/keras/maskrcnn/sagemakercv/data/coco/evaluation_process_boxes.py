def process_boxes(image_info, box_coordinates):
    """Process the model prediction for COCO eval."""
    """image_info = prediction['image_info']
    box_coordinates = prediction['detection_boxes']"""
    processed_box_coordinates = np.zeros_like(box_coordinates)
    for image_id in range(box_coordinates.shape[0]):
        scale = image_info[image_id][2]
        for box_id in range(box_coordinates.shape[1]):
            y1, x1, y2, x2 = box_coordinates[image_id, box_id, :]
            new_box = scale * np.array([x1, y1, x2 - x1, y2 - y1])
            processed_box_coordinates[image_id, box_id, :] = new_box
    return processed_box_coordinates
