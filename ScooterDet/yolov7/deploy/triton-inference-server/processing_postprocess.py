def postprocess(num_dets, det_boxes, det_scores, det_classes, img_w, img_h,
    input_shape, letter_box=True):
    boxes = det_boxes[0, :num_dets[0][0]] / np.array([input_shape[0],
        input_shape[1], input_shape[0], input_shape[1]], dtype=np.float32)
    scores = det_scores[0, :num_dets[0][0]]
    classes = det_classes[0, :num_dets[0][0]].astype(np.int)
    old_h, old_w = img_h, img_w
    offset_h, offset_w = 0, 0
    if letter_box:
        if img_w / input_shape[1] >= img_h / input_shape[0]:
            old_h = int(input_shape[0] * img_w / input_shape[1])
            offset_h = (old_h - img_h) // 2
        else:
            old_w = int(input_shape[1] * img_h / input_shape[0])
            offset_w = (old_w - img_w) // 2
    boxes = boxes * np.array([old_w, old_h, old_w, old_h], dtype=np.float32)
    if letter_box:
        boxes -= np.array([offset_w, offset_h, offset_w, offset_h], dtype=
            np.float32)
    boxes = boxes.astype(np.int)
    detected_objects = []
    for box, score, label in zip(boxes, scores, classes):
        detected_objects.append(BoundingBox(label, score, box[0], box[2],
            box[1], box[3], img_w, img_h))
    return detected_objects
