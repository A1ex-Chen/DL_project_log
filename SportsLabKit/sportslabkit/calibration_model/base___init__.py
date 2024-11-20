def __init__(self):
    """
        Initializes the base detection model.

        Args:
            model_config (Optional[dict]): The configuration for the model. This is optional and can be used to pass additional parameters to the model.
        """
    super().__init__()
    self.input_is_batched = False
