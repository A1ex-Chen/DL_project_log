def instances_to_json(self, instances, img_id):
    num_instance = len(instances)
    if num_instance == 0:
        return []
    boxes = instances.pred_boxes.tensor.numpy()
    if boxes.shape[1] == 4:
        boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    results = []
    for k in range(num_instance):
        result = {'image_id': img_id, 'category_id': classes[k], 'bbox':
            boxes[k], 'score': scores[k]}
        results.append(result)
    return results
