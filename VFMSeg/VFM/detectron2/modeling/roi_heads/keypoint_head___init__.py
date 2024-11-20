@configurable
def __init__(self, input_shape, *, num_keypoints, conv_dims, **kwargs):
    """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
        """
    super().__init__(num_keypoints=num_keypoints, **kwargs)
    up_scale = 2.0
    in_channels = input_shape.channels
    for idx, layer_channels in enumerate(conv_dims, 1):
        module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1)
        self.add_module('conv_fcn{}'.format(idx), module)
        self.add_module('conv_fcn_relu{}'.format(idx), nn.ReLU())
        in_channels = layer_channels
    deconv_kernel = 4
    self.score_lowres = ConvTranspose2d(in_channels, num_keypoints,
        deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1)
    self.up_scale = up_scale
    for name, param in self.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
