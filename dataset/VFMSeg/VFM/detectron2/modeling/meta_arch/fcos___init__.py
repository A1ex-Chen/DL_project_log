def __init__(self, *, input_shape: List[ShapeSpec], conv_dims: List[int],
    **kwargs):
    super().__init__(input_shape=input_shape, conv_dims=conv_dims,
        num_anchors=1, **kwargs)
    self._num_features = len(input_shape)
    self.ctrness = nn.Conv2d(conv_dims[-1], 1, kernel_size=3, stride=1,
        padding=1)
    torch.nn.init.normal_(self.ctrness.weight, std=0.01)
    torch.nn.init.constant_(self.ctrness.bias, 0)
