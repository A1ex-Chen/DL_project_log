def __init__(self, builder: LayerBuilder, depsep_kernel_size: int,
    in_channels: int, out_channels: int, expand_ratio: int, stride: int,
    squeeze_excitation_ratio: int, squeeze_hidden=False, survival_prob:
    float=1.0, quantized: bool=False, trt: bool=False):
    super().__init__()
    self.quantized = quantized
    self.residual = stride == 1 and in_channels == out_channels
    hidden_dim = in_channels * expand_ratio
    squeeze_base = hidden_dim if squeeze_hidden else in_channels
    squeeze_dim = max(1, int(squeeze_base * squeeze_excitation_ratio))
    self.expand = None if in_channels == hidden_dim else builder.conv1x1(
        in_channels, hidden_dim, bn=True, act=True)
    self.depsep = builder.convDepSep(depsep_kernel_size, hidden_dim,
        hidden_dim, stride, bn=True, act=True)
    self.se = SequentialSqueezeAndExcitation(hidden_dim, squeeze_dim,
        builder.activation(), self.quantized, use_conv=trt)
    self.proj = builder.conv1x1(hidden_dim, out_channels, bn=True)
    self.survival_prob = survival_prob
    if self.quantized and self.residual:
        assert quant_nn is not None, 'pytorch_quantization is not available'
        self.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.
            QuantConv2d.default_quant_desc_input)
