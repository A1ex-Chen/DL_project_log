def _tasks_from_predictions(self, predictions):
    for pred in predictions:
        if 'segmentation' in pred:
            return 'bbox', 'segm'
    return 'bbox',
