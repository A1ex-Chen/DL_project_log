@configurable
def __init__(self, input_shape: Dict[str, ShapeSpec], *,
    transformer_dropout: float, transformer_nheads: int,
    transformer_dim_feedforward: int, transformer_enc_layers: int,
    transformer_pre_norm: bool, conv_dim: int, mask_dim: int, mask_on: int,
    norm: Optional[Union[str, Callable]]=None):
    """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            transformer_pre_norm: whether to use pre-layernorm or not
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
    super().__init__(input_shape, conv_dim=conv_dim, mask_dim=mask_dim,
        norm=norm, mask_on=mask_on)
    input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
    self.in_features = [k for k, v in input_shape]
    feature_strides = [v.stride for k, v in input_shape]
    feature_channels = [v.channels for k, v in input_shape]
    in_channels = feature_channels[len(self.in_features) - 1]
    self.input_proj = Conv2d(in_channels, conv_dim, kernel_size=1)
    weight_init.c2_xavier_fill(self.input_proj)
    self.transformer = TransformerEncoderOnly(d_model=conv_dim, dropout=
        transformer_dropout, nhead=transformer_nheads, dim_feedforward=
        transformer_dim_feedforward, num_encoder_layers=
        transformer_enc_layers, normalize_before=transformer_pre_norm)
    N_steps = conv_dim // 2
    self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
    use_bias = norm == ''
    output_norm = get_norm(norm, conv_dim)
    output_conv = Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1,
        padding=1, bias=use_bias, norm=output_norm, activation=F.relu)
    weight_init.c2_xavier_fill(output_conv)
    delattr(self, 'layer_{}'.format(len(self.in_features)))
    self.add_module('layer_{}'.format(len(self.in_features)), output_conv)
    self.output_convs[0] = output_conv
