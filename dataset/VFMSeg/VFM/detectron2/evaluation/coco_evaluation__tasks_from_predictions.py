def _tasks_from_predictions(self, predictions):
    """
        Get COCO API "tasks" (i.e. iou_type) from COCO-format predictions.
        """
    tasks = {'bbox'}
    for pred in predictions:
        if 'segmentation' in pred:
            tasks.add('segm')
        if 'keypoints' in pred:
            tasks.add('keypoints')
    return sorted(tasks)
