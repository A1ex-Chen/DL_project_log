def load(self, weights: Union[str, Path]='yolov8n.pt') ->'Model':
    """
        Loads parameters from the specified weights file into the model.

        This method supports loading weights from a file or directly from a weights object. It matches parameters by
        name and shape and transfers them to the model.

        Args:
            weights (str | Path): Path to the weights file or a weights object. Defaults to 'yolov8n.pt'.

        Returns:
            self (ultralytics.engine.model.Model): The instance of the class with loaded weights.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
    self._check_is_pytorch_model()
    if isinstance(weights, (str, Path)):
        weights, self.ckpt = attempt_load_one_weight(weights)
    self.model.load(weights)
    return self
