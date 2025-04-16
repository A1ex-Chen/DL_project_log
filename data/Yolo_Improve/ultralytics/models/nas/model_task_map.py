@property
def task_map(self):
    """Returns a dictionary mapping tasks to respective predictor and validator classes."""
    return {'detect': {'predictor': NASPredictor, 'validator': NASValidator}}
