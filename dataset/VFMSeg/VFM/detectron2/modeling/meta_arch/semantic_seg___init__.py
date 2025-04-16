@configurable
def __init__(self, input_shape: Dict[str, ShapeSpec], *, num_classes: int,
    conv_dims: int, common_stride: int, loss_weight: float=1.0, norm:
    Optional[Union[str, Callable]]=None, ignore_value: int=-1):
    """
        NOTE: this interface is experimental.

        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            conv_dims: number of output channels for the intermediate conv layers.
            common_stride: the common stride that all features will be upscaled to
            loss_weight: loss weight
            norm (str or callable): normalization for all conv layers
            ignore_value: category id to be ignored during training.
        """
    super().__init__()
    input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
    if not len(input_shape):
        raise ValueError('SemSegFPNHead(input_shape=) cannot be empty!')
    self.in_features = [k for k, v in input_shape]
    feature_strides = [v.stride for k, v in input_shape]
    feature_channels = [v.channels for k, v in input_shape]
    self.ignore_value = ignore_value
    self.common_stride = common_stride
    self.loss_weight = loss_weight
    self.scale_heads = []
    for in_feature, stride, channels in zip(self.in_features,
        feature_strides, feature_channels):
        head_ops = []
        head_length = max(1, int(np.log2(stride) - np.log2(self.common_stride))
            )
        for k in range(head_length):
            norm_module = get_norm(norm, conv_dims)
            conv = Conv2d(channels if k == 0 else conv_dims, conv_dims,
                kernel_size=3, stride=1, padding=1, bias=not norm, norm=
                norm_module, activation=F.relu)
            weight_init.c2_msra_fill(conv)
            head_ops.append(conv)
            if stride != self.common_stride:
                head_ops.append(nn.Upsample(scale_factor=2, mode='bilinear',
                    align_corners=False))
        self.scale_heads.append(nn.Sequential(*head_ops))
        self.add_module(in_feature, self.scale_heads[-1])
    self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1,
        padding=0)
    weight_init.c2_msra_fill(self.predictor)
