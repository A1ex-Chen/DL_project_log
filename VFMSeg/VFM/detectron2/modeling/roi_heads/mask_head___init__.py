@configurable
def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims,
    conv_norm='', **kwargs):
    """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_classes (int): the number of foreground classes (i.e. background is not
                included). 1 if using class agnostic prediction.
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
    super().__init__(**kwargs)
    assert len(conv_dims) >= 1, 'conv_dims have to be non-empty!'
    self.conv_norm_relus = []
    cur_channels = input_shape.channels
    for k, conv_dim in enumerate(conv_dims[:-1]):
        conv = Conv2d(cur_channels, conv_dim, kernel_size=3, stride=1,
            padding=1, bias=not conv_norm, norm=get_norm(conv_norm,
            conv_dim), activation=nn.ReLU())
        self.add_module('mask_fcn{}'.format(k + 1), conv)
        self.conv_norm_relus.append(conv)
        cur_channels = conv_dim
    self.deconv = ConvTranspose2d(cur_channels, conv_dims[-1], kernel_size=
        2, stride=2, padding=0)
    self.add_module('deconv_relu', nn.ReLU())
    cur_channels = conv_dims[-1]
    self.predictor = Conv2d(cur_channels, num_classes, kernel_size=1,
        stride=1, padding=0)
    for layer in (self.conv_norm_relus + [self.deconv]):
        weight_init.c2_msra_fill(layer)
    nn.init.normal_(self.predictor.weight, std=0.001)
    if self.predictor.bias is not None:
        nn.init.constant_(self.predictor.bias, 0)
