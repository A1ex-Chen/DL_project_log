@configurable
def __init__(self, *, input_shape: List[ShapeSpec], num_classes,
    num_anchors, conv_dims: List[int], norm='', prior_prob=0.01):
    """
        NOTE: this interface is experimental.

        Args:
            input_shape (List[ShapeSpec]): input shape
            num_classes (int): number of classes. Used to label background proposals.
            num_anchors (int): number of generated anchors
            conv_dims (List[int]): dimensions for each convolution layer
            norm (str or callable):
                Normalization for conv layers except for the two output layers.
                See :func:`detectron2.layers.get_norm` for supported types.
            prior_prob (float): Prior weight for computing bias
        """
    super().__init__()
    self._num_features = len(input_shape)
    if norm == 'BN' or norm == 'SyncBN':
        logger.info(
            f'Using domain-specific {norm} in RetinaNetHead with len={self._num_features}.'
            )
        bn_class = nn.BatchNorm2d if norm == 'BN' else nn.SyncBatchNorm

        def norm(c):
            return CycleBatchNormList(length=self._num_features, bn_class=
                bn_class, num_features=c)
    else:
        norm_name = str(type(get_norm(norm, 1)))
        if 'BN' in norm_name:
            logger.warning(
                f'Shared BatchNorm (type={norm_name}) may not work well in RetinaNetHead.'
                )
    cls_subnet = []
    bbox_subnet = []
    for in_channels, out_channels in zip([input_shape[0].channels] + list(
        conv_dims), conv_dims):
        cls_subnet.append(nn.Conv2d(in_channels, out_channels, kernel_size=
            3, stride=1, padding=1))
        if norm:
            cls_subnet.append(get_norm(norm, out_channels))
        cls_subnet.append(nn.ReLU())
        bbox_subnet.append(nn.Conv2d(in_channels, out_channels, kernel_size
            =3, stride=1, padding=1))
        if norm:
            bbox_subnet.append(get_norm(norm, out_channels))
        bbox_subnet.append(nn.ReLU())
    self.cls_subnet = nn.Sequential(*cls_subnet)
    self.bbox_subnet = nn.Sequential(*bbox_subnet)
    self.cls_score = nn.Conv2d(conv_dims[-1], num_anchors * num_classes,
        kernel_size=3, stride=1, padding=1)
    self.bbox_pred = nn.Conv2d(conv_dims[-1], num_anchors * 4, kernel_size=
        3, stride=1, padding=1)
    for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self
        .bbox_pred]:
        for layer in modules.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)
    bias_value = -math.log((1 - prior_prob) / prior_prob)
    torch.nn.init.constant_(self.cls_score.bias, bias_value)
