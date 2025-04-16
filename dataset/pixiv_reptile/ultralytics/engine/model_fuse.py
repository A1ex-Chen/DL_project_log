def fuse(self):
    """
        Fuses Conv2d and BatchNorm2d layers in the model.

        This method optimizes the model by fusing Conv2d and BatchNorm2d layers, which can improve inference speed.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
    self._check_is_pytorch_model()
    self.model.fuse()
