@configurable
def __init__(self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims:
    List[int], conv_norm=''):
    """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
    super().__init__()
    assert len(conv_dims) + len(fc_dims) > 0
    self._output_size = (input_shape.channels, input_shape.height,
        input_shape.width)
    self.conv_norm_relus = []
    for k, conv_dim in enumerate(conv_dims):
        conv = Conv2d(self._output_size[0], conv_dim, kernel_size=3,
            padding=1, bias=not conv_norm, norm=get_norm(conv_norm,
            conv_dim), activation=nn.ReLU())
        self.add_module('conv{}'.format(k + 1), conv)
        self.conv_norm_relus.append(conv)
        self._output_size = conv_dim, self._output_size[1], self._output_size[2
            ]
    self.fcs = []
    for k, fc_dim in enumerate(fc_dims):
        if k == 0:
            self.add_module('flatten', nn.Flatten())
        fc = nn.Linear(int(np.prod(self._output_size)), fc_dim)
        self.add_module('fc{}'.format(k + 1), fc)
        self.add_module('fc_relu{}'.format(k + 1), nn.ReLU())
        self.fcs.append(fc)
        self._output_size = fc_dim
    for layer in self.conv_norm_relus:
        weight_init.c2_msra_fill(layer)
    for layer in self.fcs:
        weight_init.c2_xavier_fill(layer)
