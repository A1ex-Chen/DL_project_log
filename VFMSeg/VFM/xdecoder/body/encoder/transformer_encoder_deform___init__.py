@configurable
def __init__(self, input_shape: Dict[str, ShapeSpec], *,
    transformer_dropout: float, transformer_nheads: int,
    transformer_dim_feedforward: int, transformer_enc_layers: int, conv_dim:
    int, mask_dim: int, norm: Optional[Union[str, Callable]]=None,
    transformer_in_features: List[str], common_stride: int):
    """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
    super().__init__()
    transformer_input_shape = {k: v for k, v in input_shape.items() if k in
        transformer_in_features}
    input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
    self.in_features = [k for k, v in input_shape]
    self.feature_strides = [v.stride for k, v in input_shape]
    self.feature_channels = [v.channels for k, v in input_shape]
    transformer_input_shape = sorted(transformer_input_shape.items(), key=
        lambda x: x[1].stride)
    self.transformer_in_features = [k for k, v in transformer_input_shape]
    transformer_in_channels = [v.channels for k, v in transformer_input_shape]
    self.transformer_feature_strides = [v.stride for k, v in
        transformer_input_shape]
    self.transformer_num_feature_levels = len(self.transformer_in_features)
    if self.transformer_num_feature_levels > 1:
        input_proj_list = []
        for in_channels in transformer_in_channels[::-1]:
            input_proj_list.append(nn.Sequential(nn.Conv2d(in_channels,
                conv_dim, kernel_size=1), nn.GroupNorm(32, conv_dim)))
        self.input_proj = nn.ModuleList(input_proj_list)
    else:
        self.input_proj = nn.ModuleList([nn.Sequential(nn.Conv2d(
            transformer_in_channels[-1], conv_dim, kernel_size=1), nn.
            GroupNorm(32, conv_dim))])
    for proj in self.input_proj:
        nn.init.xavier_uniform_(proj[0].weight, gain=1)
        nn.init.constant_(proj[0].bias, 0)
    self.transformer = MSDeformAttnTransformerEncoderOnly(d_model=conv_dim,
        dropout=transformer_dropout, nhead=transformer_nheads,
        dim_feedforward=transformer_dim_feedforward, num_encoder_layers=
        transformer_enc_layers, num_feature_levels=self.
        transformer_num_feature_levels)
    N_steps = conv_dim // 2
    self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
    self.mask_dim = mask_dim
    self.mask_features = Conv2d(conv_dim, mask_dim, kernel_size=1, stride=1,
        padding=0)
    weight_init.c2_xavier_fill(self.mask_features)
    self.maskformer_num_feature_levels = 3
    self.common_stride = common_stride
    stride = min(self.transformer_feature_strides)
    self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))
    lateral_convs = []
    output_convs = []
    use_bias = norm == ''
    for idx, in_channels in enumerate(self.feature_channels[:self.
        num_fpn_levels]):
        lateral_norm = get_norm(norm, conv_dim)
        output_norm = get_norm(norm, conv_dim)
        lateral_conv = Conv2d(in_channels, conv_dim, kernel_size=1, bias=
            use_bias, norm=lateral_norm)
        output_conv = Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1,
            padding=1, bias=use_bias, norm=output_norm, activation=F.relu)
        weight_init.c2_xavier_fill(lateral_conv)
        weight_init.c2_xavier_fill(output_conv)
        self.add_module('adapter_{}'.format(idx + 1), lateral_conv)
        self.add_module('layer_{}'.format(idx + 1), output_conv)
        lateral_convs.append(lateral_conv)
        output_convs.append(output_conv)
    self.lateral_convs = lateral_convs[::-1]
    self.output_convs = output_convs[::-1]
