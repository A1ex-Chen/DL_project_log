def __init__(self, length: int, bn_class=nn.BatchNorm2d, **kwargs):
    """
        Args:
            length: number of BatchNorm layers to cycle.
            bn_class: the BatchNorm class to use
            kwargs: arguments of the BatchNorm class, such as num_features.
        """
    self._affine = kwargs.pop('affine', True)
    super().__init__([bn_class(**kwargs, affine=False) for k in range(length)])
    if self._affine:
        channels = self[0].num_features
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
    self._pos = 0
