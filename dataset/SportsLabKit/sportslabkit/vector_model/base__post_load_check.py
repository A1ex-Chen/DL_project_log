def _post_load_check(self) ->None:
    """Check whether the model has been loaded correctly.
        """
    if self.model is None:
        raise ValueError(
            'Model not loaded correctly. Fix your _load_model implementation.')
