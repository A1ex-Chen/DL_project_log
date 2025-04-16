def __init__(self, *args, **kwargs):
    """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
    norm = kwargs.pop('norm', None)
    activation = kwargs.pop('activation', None)
    super().__init__(*args, **kwargs)
    self.norm = norm
    self.activation = activation
