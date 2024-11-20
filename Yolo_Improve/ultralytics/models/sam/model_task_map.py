@property
def task_map(self):
    """
        Provides a mapping from the 'segment' task to its corresponding 'Predictor'.

        Returns:
            (dict): A dictionary mapping the 'segment' task to its corresponding 'Predictor'.
        """
    return {'segment': {'predictor': Predictor}}
