def prepare_for_coco_detection(self, predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue
        boxes = prediction['boxes']
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction['scores'].tolist()
        labels = prediction['labels'].tolist()
        coco_results.extend([{'image_id': original_id, 'category_id':
            labels[k], 'bbox': box, 'score': scores[k]} for k, box in
            enumerate(boxes)])
    return coco_results
