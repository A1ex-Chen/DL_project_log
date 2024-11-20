@property
def task_map(self) ->dict:
    """
        Map head to model, trainer, validator, and predictor classes.

        Returns:
            task_map (dict): The map of model task to mode classes.
        """
    raise NotImplementedError('Please provide task map for your model!')
