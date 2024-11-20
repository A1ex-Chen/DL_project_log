@property
def task_map(self):
    """Returns a dictionary mapping segment task to corresponding predictor and validator classes."""
    return {'segment': {'predictor': FastSAMPredictor, 'validator':
        FastSAMValidator}}
