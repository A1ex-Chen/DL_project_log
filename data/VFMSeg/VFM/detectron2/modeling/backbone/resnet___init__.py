def __init__(self, stem, stages, num_classes=None, out_features=None,
    freeze_at=0):
    """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[CNNBlockBase]]): several (typically 4) stages,
                each contains multiple :class:`CNNBlockBase`.
            num_classes (None or int): if None, will not perform classification.
                Otherwise, will create a linear layer.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
            freeze_at (int): The number of stages at the beginning to freeze.
                see :meth:`freeze` for detailed explanation.
        """
    super().__init__()
    self.stem = stem
    self.num_classes = num_classes
    current_stride = self.stem.stride
    self._out_feature_strides = {'stem': current_stride}
    self._out_feature_channels = {'stem': self.stem.out_channels}
    self.stage_names, self.stages = [], []
    if out_features is not None:
        num_stages = max([{'res2': 1, 'res3': 2, 'res4': 3, 'res5': 4}.get(
            f, 0) for f in out_features])
        stages = stages[:num_stages]
    for i, blocks in enumerate(stages):
        assert len(blocks) > 0, len(blocks)
        for block in blocks:
            assert isinstance(block, CNNBlockBase), block
        name = 'res' + str(i + 2)
        stage = nn.Sequential(*blocks)
        self.add_module(name, stage)
        self.stage_names.append(name)
        self.stages.append(stage)
        self._out_feature_strides[name] = current_stride = int(
            current_stride * np.prod([k.stride for k in blocks]))
        self._out_feature_channels[name] = curr_channels = blocks[-1
            ].out_channels
    self.stage_names = tuple(self.stage_names)
    if num_classes is not None:
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(curr_channels, num_classes)
        nn.init.normal_(self.linear.weight, std=0.01)
        name = 'linear'
    if out_features is None:
        out_features = [name]
    self._out_features = out_features
    assert len(self._out_features)
    children = [x[0] for x in self.named_children()]
    for out_feature in self._out_features:
        assert out_feature in children, 'Available children: {}'.format(', '
            .join(children))
    self.freeze(freeze_at)
