@property
def task_map(self):
    """Map head to model, validator, and predictor classes."""
    return {'detect': {'model': WorldModel, 'validator': yolo.detect.
        DetectionValidator, 'predictor': yolo.detect.DetectionPredictor,
        'trainer': yolo.world.WorldTrainer}}
