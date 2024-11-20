def prepare_for_coco_keypoint(self, predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue
        boxes = prediction['boxes']
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction['scores'].tolist()
        labels = prediction['labels'].tolist()
        keypoints = prediction['keypoints']
        keypoints = keypoints.flatten(start_dim=1).tolist()
        coco_results.extend([{'image_id': original_id, 'category_id':
            labels[k], 'keypoints': keypoint, 'score': scores[k]} for k,
            keypoint in enumerate(keypoints)])
    return coco_results
