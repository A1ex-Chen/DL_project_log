@property
def task_map(self) ->dict:
    """
        Returns a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

        Returns:
            dict: A dictionary mapping task names to Ultralytics task classes for the RT-DETR model.
        """
    return {'detect': {'predictor': RTDETRPredictor, 'validator':
        RTDETRValidator, 'trainer': RTDETRTrainer, 'model':
        RTDETRDetectionModel}}
